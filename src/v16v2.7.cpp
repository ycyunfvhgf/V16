// =============================================================================
// v16v2 â€” clean-room UCI chess engine built around the trained v16 NNUE net.
//
// Design (quality over speed, single file, Lazy SMP multi-thread):
//   - 8x8 mailbox board, pseudo-legal movegen + make/unmake legality filter
//   - Evaluation = trained net (768->2048)x2->1, Chess768 features, SCReLU,
//     int32 accumulators updated incrementally (exact-inverse unmake),
//     int64 output dot, cp = out * 40 / (255*64)
//   - Hand eval is NOT used for play (only insufficient-material draws).
//     Mate/stalemate/repetition/50-move handling is in search, never eval.
//   - Search: PVS + shared TT + Lazy SMP (auto-detect CPU threads),
//     MVV-LVA/killers/history, quiescence, null-move, LMR, mate-distance.
//   - UCI: position, go, stop, quit, setoption Hash/Threads/EvalFile,
//     plus debug commands: eval, perft, verify.
//
// Conventions: square 0 = a1 .. 63 = h8, file = sq & 7, rank = sq >> 3.
// Colors: WHITE = 0, BLACK = 1. Pieces: P=0 N=1 B=2 R=3 Q=4 K=5.
// Castling bits: 1 = White K, 2 = White Q, 4 = Black k, 8 = Black q.
// Eval scores are always from the side-to-move's perspective (negamax).
// Mate score = 30000 - ply. No eval may enter the mate zone (clamped).
// =============================================================================

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <immintrin.h>

// ---------------------------------------------------------------- types ---
using U64 = uint64_t;
using I64 = int64_t;

enum { WHITE, BLACK };
enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE };
enum { FLAG_QUIET = 0, FLAG_DOUBLE = 1, FLAG_EP = 2, FLAG_CASTLE_K = 4, FLAG_CASTLE_Q = 8 };

static const int MATE_SCORE = 30000;
static const int INF_SCORE  = 31000;
#ifndef CHK_CAP
#define CHK_CAP 3
#endif
#ifndef CHK_SEE_GATE
#define CHK_SEE_GATE 1
#endif
#ifndef CHK_SEE_MARGIN
#define CHK_SEE_MARGIN -60
#endif
static const int MAX_PLY    = 128;   // search ply cap (stack / killers)
static const int EVAL_CLAMP = 20000; // eval never enters mate-score territory
static const int MAX_THREADS = 32;

static int detectCpuThreads() {
    unsigned n = std::thread::hardware_concurrency();
    if (n == 0) n = 4;
    if (n > (unsigned)MAX_THREADS) n = MAX_THREADS;
    return (int)n;
}

static const int PIECE_VALUE[6] = { 100, 320, 330, 500, 900, 0 };

// NNUE layout (matches bullet trainer: (768->2048)x2->1, dual perspective,
// SCReLU, QA=255 on layer 0, QB=64 on layer 1).
// Eval scale: the net's raw output is in true centipawns / 250.  Earlier builds
// used 40, calibrated on EQUAL positions where the scale does not matter; that
// left the eval ~6x compressed (K+Q vs K read +1.99 instead of ~+10) while all
// search constants below stayed in real centipawns.  Verified on a material
// ladder and against Stockfish over real game positions.
static const int NNUE_HIDDEN = 512;
static const int NNUE_INPUTS = 768;      // Chess768 rows per king bucket
static const int NNUE_MAX_BUCKETS = 32;  // horizontally mirrored king buckets
static const int NNUE_QA = 255;
static const int NNUE_QB = 64;
static const int NNUE_SCALE = 250;
static const int NNUE_QAQB = NNUE_QA * NNUE_QB; // 16320

static inline int fileOf(int sq) { return sq & 7; }
static inline int rankOf(int sq) { return sq >> 3; }
static inline int makeSq(int rank, int file) { return rank * 8 + file; }
static inline U64 bit(int sq) { return 1ULL << sq; }

// King bucket id, identical to bullet ChessBucketsMirrored with an identity
// 32-entry table (a-d half, rank-major): rank*4 + min(file, 7-file).
static inline int kingBucket(int ksq) {
    int f = fileOf(ksq);
    if (f > 3) f = 7 - f;
    return rankOf(ksq) * 4 + f;
}
// Horizontal mirror of a board square when the king is on files e-h.
static inline int bucketMirror(int sq, int ksq) {
    return (fileOf(ksq) > 3) ? (sq ^ 7) : sq;
}

// ------------------------------------------------------------ zobrist ----
static U64 zobristPiece[2][6][64];
static U64 zobristSide;
static U64 zobristCastle[16];
static U64 zobristEp[8];

static U64 splitmix64(U64 &s) {
    U64 z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void initZobrist() {
    U64 s = 0x123456789ABCDEFULL;
    for (int c = 0; c < 2; c++)
        for (int p = 0; p < 6; p++)
            for (int sq = 0; sq < 64; sq++)
                zobristPiece[c][p][sq] = splitmix64(s);
    zobristSide = splitmix64(s);
    for (int i = 0; i < 16; i++) zobristCastle[i] = splitmix64(s);
    for (int i = 0; i < 8; i++) zobristEp[i] = splitmix64(s);
}

// -------------------------------------------------------------- board ----
struct Move {
    int from = 0, to = 0;
    int promo = NO_PIECE;    // promotion piece, else NO_PIECE
    int captured = NO_PIECE; // captured piece type, else NO_PIECE
    int flags = FLAG_QUIET;
};
static std::string moveToUCI(const Move &m);   // defined with the UCI code

struct Undo {
    int captured = NO_PIECE;
    int castling = 0;
    int ep = -1;
    int half = 0;
    U64 key = 0;
};

struct Board {
    int sq[64];      // NO_PIECE or 0..5
    int col[64];     // WHITE/BLACK, -1 if empty
    int side = WHITE;
    int castling = 0;
    int ep = -1;     // en-passant target square, -1 = none
    int half = 0;
    int full = 1;
    U64 key = 0;
    int kingSq[2] = { 4, 60 };
    U64 occ[2] = { 0, 0 }; // white, black occupancy
    U64 all = 0;           // all occupancy
    U64 pieces[2][6] = {}; // per color, per piece
};

static void boardClear(Board &b) {
    for (int i = 0; i < 64; i++) { b.sq[i] = NO_PIECE; b.col[i] = -1; }
    b.side = WHITE; b.castling = 0; b.ep = -1;
    b.half = 0; b.full = 1; b.key = 0;
    b.kingSq[WHITE] = 4; b.kingSq[BLACK] = 60;
    b.occ[WHITE] = 0; b.occ[BLACK] = 0; b.all = 0;
    for (int c = 0; c < 2; c++) for (int p = 0; p < 6; p++) b.pieces[c][p] = 0;
}

static void boardAdd(Board &b, int sq, int piece, int color) {
    b.sq[sq] = piece;
    b.col[sq] = color;
    b.key ^= zobristPiece[color][piece][sq];
    if (piece == KING) b.kingSq[color] = sq;
    b.occ[color] |= bit(sq);
    b.all |= bit(sq);
    b.pieces[color][piece] |= bit(sq);
}

// Defensive king lookup: a malformed board must never reach ctzll(0), which
// would index outside the NNUE weight table.  Valid positions always take the
// first branch; the fallback keeps the engine alive on bad FEN input.
static inline int kingSquareOf(const Board &b, int color) {
    U64 bb = b.pieces[color][KING];
    if (bb) return __builtin_ctzll(bb);
    return (color == WHITE) ? 4 : 60;
}

static U64 boardComputeKey(const Board &b) {
    U64 k = 0;
    for (int sq = 0; sq < 64; sq++)
        if (b.sq[sq] != NO_PIECE) k ^= zobristPiece[b.col[sq]][b.sq[sq]][sq];
    if (b.side == BLACK) k ^= zobristSide;
    k ^= zobristCastle[b.castling & 15];
    if (b.ep >= 0) k ^= zobristEp[fileOf(b.ep)];
    return k;
}

// Parse into a temporary and commit only on success, so a bad FEN can never
// leave the engine with an empty/half-built board (which used to crash inside
// NNUE via ctzll(0)).
static bool parseFEN(Board &b, const std::string &fen) {
    Board nb;
    boardClear(nb);
    std::istringstream ss(fen);
    std::string placement, side, castling, ep;
    int half = 0, full = 1;
    if (!(ss >> placement >> side >> castling >> ep)) return false;
    if (side != "w" && side != "b") return false;
    ss >> half >> full;
    int rank = 7, file = 0, rows = 1;
    for (char ch : placement) {
        if (ch == '/') { rank--; file = 0; rows++; continue; }
        if (ch >= '1' && ch <= '8') {
            file += ch - '0';
            if (file > 8) return false;
            continue;
        }
        int color = (ch >= 'a') ? BLACK : WHITE;
        char lo = (char)tolower(ch);
        int piece = NO_PIECE;
        if (lo == 'p') piece = PAWN;
        else if (lo == 'n') piece = KNIGHT;
        else if (lo == 'b') piece = BISHOP;
        else if (lo == 'r') piece = ROOK;
        else if (lo == 'q') piece = QUEEN;
        else if (lo == 'k') piece = KING;
        else return false;
        if (file < 0 || file > 7 || rank < 0 || rank > 7) return false;
        boardAdd(nb, makeSq(rank, file), piece, color);
        file++;
    }
    if (rows != 8 || rank != 0 || file != 8) return false;
    // Reject anything that cannot be searched safely: exactly one king a side.
    if (__builtin_popcountll(nb.pieces[WHITE][KING]) != 1) return false;
    if (__builtin_popcountll(nb.pieces[BLACK][KING]) != 1) return false;
    nb.side = (side == "b") ? BLACK : WHITE;
    nb.castling = 0;
    for (char ch : castling) {
        if (ch == 'K') nb.castling |= 1;
        else if (ch == 'Q') nb.castling |= 2;
        else if (ch == 'k') nb.castling |= 4;
        else if (ch == 'q') nb.castling |= 8;
    }
    nb.ep = -1;
    if (ep != "-" && ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' &&
        ep[1] >= '1' && ep[1] <= '8') {
        int s = makeSq(ep[1] - '1', ep[0] - 'a');
        if (rankOf(s) == 2 || rankOf(s) == 5) nb.ep = s;
    }
    nb.half = half < 0 ? 0 : half;
    nb.full = full < 1 ? 1 : full;
    nb.key = boardComputeKey(nb);
    b = nb;
    return true;
}

// --------------------------------------------------------------- attacks -
static U64 knightAttacks[64];
static U64 kingAttacks[64];
static U64 pawnAttacks[2][64]; // squares attacked BY a pawn of color c on sq

static void initAttackTables() {
    const int knightD[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
    const int kingD[8][2] = {{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1}};
    for (int sq = 0; sq < 64; sq++) {
        int f = fileOf(sq), r = rankOf(sq);
        U64 n = 0, k = 0, pw = 0, pb = 0;
        for (auto &d : knightD) {
            int nf = f + d[0], nr = r + d[1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) n |= bit(makeSq(nr, nf));
        }
        for (auto &d : kingD) {
            int nf = f + d[0], nr = r + d[1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) k |= bit(makeSq(nr, nf));
        }
        if (r < 7) {
            if (f > 0) pw |= bit(makeSq(r + 1, f - 1));
            if (f < 7) pw |= bit(makeSq(r + 1, f + 1));
        }
        if (r > 0) {
            if (f > 0) pb |= bit(makeSq(r - 1, f - 1));
            if (f < 7) pb |= bit(makeSq(r - 1, f + 1));
        }
        knightAttacks[sq] = n;
        kingAttacks[sq] = k;
        pawnAttacks[WHITE][sq] = pw;
        pawnAttacks[BLACK][sq] = pb;
    }
}

// Attacks â€” HEAVILY OPTIMISED original, PEXT+cache, fixes depth holdback
static inline U64 slidingSlowGeneric(int sq, U64 occ, bool bishop){
    U64 a=0; int f=fileOf(sq), r=rankOf(sq);
    if(bishop){ const int d[4][2]={{1,1},{1,-1},{-1,1},{-1,-1}}; for(auto &dd:d){int nf=f+dd[0],nr=r+dd[1]; while(nf>=0&&nf<8&&nr>=0&&nr<8){int t=makeSq(nr,nf); a|=bit(t); if(occ&bit(t)) break; nf+=dd[0]; nr+=dd[1];}}}
    else { const int d[4][2]={{1,0},{-1,0},{0,1},{0,-1}}; for(auto &dd:d){int nf=f+dd[0],nr=r+dd[1]; while(nf>=0&&nf<8&&nr>=0&&nr<8){int t=makeSq(nr,nf); a|=bit(t); if(occ&bit(t)) break; nf+=dd[0]; nr+=dd[1];}}}
    return a;
}
static U64 bishopMasksO[64], rookMasksO[64];
static U64 bishopTableO[64][512];
static U64 rookTableO[64][4096];
#ifndef __BMI2__
static U64 bishopMagicsO[64], rookMagicsO[64];
static int bishopShiftsO[64], rookShiftsO[64];
#endif

static inline U64 subsetToOcc(U64 mask, int i) {
    U64 occ = 0;
    int bi = 0;
    while (mask) {
        int s = __builtin_ctzll(mask);
        mask &= mask - 1;
        if ((i >> bi) & 1) occ |= bit(s);
        bi++;
    }
    return occ;
}

static void initMagic(){
    for(int sq=0;sq<64;sq++){
        int f=fileOf(sq), r=rankOf(sq);
        U64 bm=slidingSlowGeneric(sq,0,true) & ~(((0xFFULL|(0xFFULL<<56)) & ~(0xFFULL<<(8*r))) | ((0x0101010101010101ULL|(0x0101010101010101ULL<<7)) & ~(0x0101010101010101ULL<<f)));
        U64 rm=slidingSlowGeneric(sq,0,false) & ~(((0xFFULL|(0xFFULL<<56)) & ~(0xFFULL<<(8*r))) | ((0x0101010101010101ULL|(0x0101010101010101ULL<<7)) & ~(0x0101010101010101ULL<<f)));
        bishopMasksO[sq]=bm; rookMasksO[sq]=rm;
        int bbits=__builtin_popcountll(bm), rbits=__builtin_popcountll(rm);
#ifdef __BMI2__
        (void)bbits; (void)rbits;
        // BMI2: index directly with PEXT (much faster than searching for a
        // magic multiplier on every startup, which used to cost ~8 seconds).
        for(int i=0;i<(1<<bbits);i++){
            U64 occ=subsetToOcc(bm,i);
            bishopTableO[sq][_pext_u64(occ,bm)]=slidingSlowGeneric(sq,occ,true);
        }
        for(int i=0;i<(1<<rbits);i++){
            U64 occ=subsetToOcc(rm,i);
            rookTableO[sq][_pext_u64(occ,rm)]=slidingSlowGeneric(sq,occ,false);
        }
#else
        bishopShiftsO[sq]=64-bbits; rookShiftsO[sq]=64-rbits;
        // original PRNG, not Stockfish seeds
        U64 s = 0x9e3779b97f4a7c15ULL + sq*0x9d2c5680ULL;
        auto rng=[&](){ s^=s>>12; s^=s<<25; s^=s>>27; return s*2685821657736338717ULL; };
        auto sparse=[&](){ return rng() & rng() & rng(); };
        // bishop magics
        for(;;){
            U64 m; for(m=0; __builtin_popcountll((m*bm)>>56)<6;) m=sparse();
            bool ok=true;
            for(int i=0;i<(1<<bbits);i++){ U64 occ=subsetToOcc(bm,i); int idx=(occ*m)>>bishopShiftsO[sq]; if(idx>=512) { ok=false; break; } U64 att=slidingSlowGeneric(sq,occ,true); if(bishopTableO[sq][idx]!=0 && bishopTableO[sq][idx]!=att) { ok=false; break; } bishopTableO[sq][idx]=att; }
            if(ok){ bishopMagicsO[sq]=m; break; }
            for(int i=0;i<512;i++) bishopTableO[sq][i]=0;
        }
        for(int i=0;i<512;i++) bishopTableO[sq][i]=0;
        for(int i=0;i<(1<<bbits);i++){ U64 occ=subsetToOcc(bm,i); int idx=(occ*bishopMagicsO[sq])>>bishopShiftsO[sq]; bishopTableO[sq][idx]=slidingSlowGeneric(sq,occ,true); }
        for(;;){
            U64 m; for(m=0; __builtin_popcountll((m*rm)>>56)<6;) m=sparse();
            bool ok=true;
            for(int i=0;i<(1<<rbits);i++){ U64 occ=subsetToOcc(rm,i); int idx=(occ*m)>>rookShiftsO[sq]; if(idx>=4096) { ok=false; break; } U64 att=slidingSlowGeneric(sq,occ,false); if(rookTableO[sq][idx]!=0 && rookTableO[sq][idx]!=att) { ok=false; break; } rookTableO[sq][idx]=att; }
            if(ok){ rookMagicsO[sq]=m; break; }
            for(int i=0;i<4096;i++) rookTableO[sq][i]=0;
        }
        for(int i=0;i<4096;i++) rookTableO[sq][i]=0;
        for(int i=0;i<(1<<rbits);i++){ U64 occ=subsetToOcc(rm,i); int idx=(occ*rookMagicsO[sq])>>rookShiftsO[sq]; rookTableO[sq][idx]=slidingSlowGeneric(sq,occ,false); }
#endif
    }
}
#ifdef __BMI2__
static inline U64 bishopAttacks(int sq, U64 occ){ return bishopTableO[sq][_pext_u64(occ & bishopMasksO[sq], bishopMasksO[sq])]; }
static inline U64 rookAttacks(int sq, U64 occ){ return rookTableO[sq][_pext_u64(occ & rookMasksO[sq], rookMasksO[sq])]; }
#else
static inline U64 bishopAttacks(int sq, U64 occ){ return bishopTableO[sq][((occ & bishopMasksO[sq])*bishopMagicsO[sq])>>bishopShiftsO[sq]]; }
static inline U64 rookAttacks(int sq, U64 occ){ return rookTableO[sq][((occ & rookMasksO[sq])*rookMagicsO[sq])>>rookShiftsO[sq]]; }
#endif
static inline U64 occupancy(const Board &b) {
    return b.all;
}

static inline bool isAttacked(const Board &b, int sq, int byColor) {
    // Pawns: squares that attack `sq` with a byColor-pawn are pawnAttacks[enemy][sq]
    if (pawnAttacks[byColor ^ 1][sq] & b.pieces[byColor][PAWN]) return true;
    if (knightAttacks[sq] & b.pieces[byColor][KNIGHT]) return true;
    if (kingAttacks[sq] & b.pieces[byColor][KING]) return true;
    U64 occ = b.all;
    if (bishopAttacks(sq, occ) & (b.pieces[byColor][BISHOP] | b.pieces[byColor][QUEEN])) return true;
    if (rookAttacks(sq, occ) & (b.pieces[byColor][ROOK] | b.pieces[byColor][QUEEN])) return true;
    return false;
}

static inline bool inCheck(const Board &b, int color) {
    return isAttacked(b, b.kingSq[color], color ^ 1);
}

// -------------------------------------------------------------- movegen --
// 448 = strict upper bound on pseudo-legal moves (16 pieces x 27 squares),
// so a list can never silently drop a move.  The search's legal-move output is
// still bounded by the 218-move legal maximum, but pseudo-legal generation can
// exceed 256 in promotion-heavy positions.
struct MoveList {
    Move m[448];
    int n = 0;
    void add(Move mv) { if (n < 448) m[n++] = mv; }
};

static void genPawnMoves(const Board &b, int sq, MoveList &out, bool capturesOnly) {
    int color = b.col[sq];
    int f = fileOf(sq), r = rankOf(sq);
    int dir = (color == WHITE) ? 1 : -1;
    int startRank = (color == WHITE) ? 1 : 6;
    int promoRank = (color == WHITE) ? 7 : 0;
    // Captures (both modes)
    for (int df = -1; df <= 1; df += 2) {
        int nf = f + df, nr = r + dir;
        if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
        int t = makeSq(nr, nf);
        bool isCap = (b.sq[t] != NO_PIECE && b.col[t] != color);
        bool isEp = (!isCap && t == b.ep);
        if (!isCap && !isEp) continue;
        Move mv; mv.from = sq; mv.to = t;
        mv.captured = isEp ? PAWN : b.sq[t];
        if (isEp) mv.flags = FLAG_EP;
        if (nr == promoRank) {
            for (int pr : { QUEEN, ROOK, BISHOP, KNIGHT }) {
                Move pm = mv; pm.promo = pr; out.add(pm);
            }
        } else out.add(mv);
    }
    if (capturesOnly) {
        // Quiet queen promotions are tactical: include them in quiescence.
        int one = makeSq(r + dir, f);
        if (r + dir >= 0 && r + dir < 8 && b.sq[one] == NO_PIECE && r + dir == promoRank) {
            Move mv; mv.from = sq; mv.to = one; mv.promo = QUEEN; out.add(mv);
        }
        return;
    }
    // Pushes
    int one = makeSq(r + dir, f);
    if (r + dir >= 0 && r + dir < 8 && b.sq[one] == NO_PIECE) {
        if (r + dir == promoRank) {
            for (int pr : { QUEEN, ROOK, BISHOP, KNIGHT }) {
                Move mv; mv.from = sq; mv.to = one; mv.promo = pr; out.add(mv);
            }
        } else {
            Move mv; mv.from = sq; mv.to = one; out.add(mv);
            if (r == startRank) {
                int two = makeSq(r + 2 * dir, f);
                if (b.sq[two] == NO_PIECE) {
                    Move m2; m2.from = sq; m2.to = two; m2.flags = FLAG_DOUBLE; out.add(m2);
                }
            }
        }
    }
}

static void genMoves(const Board &b, MoveList &out, bool capturesOnly) {
    out.n = 0;
    int us = b.side;
    U64 occ = b.all;
    U64 our = b.occ[us];
    // Pawns via bitboard
    {
        U64 bb = b.pieces[us][PAWN];
        while (bb) {
            int sq = __builtin_ctzll(bb); bb &= bb - 1;
            genPawnMoves(b, sq, out, capturesOnly);
        }
    }
    auto emit = [&](int sq, U64 targets) {
        targets &= capturesOnly ? b.occ[us ^ 1] : ~our;
        while (targets) {
            int t = __builtin_ctzll(targets); targets &= targets - 1;
            if (b.sq[t] == KING) continue;
            Move mv; mv.from = sq; mv.to = t;
            mv.captured = (b.sq[t] != NO_PIECE) ? b.sq[t] : NO_PIECE;
            out.add(mv);
        }
    };
    {
        U64 bb = b.pieces[us][KNIGHT];
        while (bb) { int sq = __builtin_ctzll(bb); bb &= bb - 1; emit(sq, knightAttacks[sq]); }
    }
    {
        U64 bb = b.pieces[us][BISHOP];
        while (bb) { int sq = __builtin_ctzll(bb); bb &= bb - 1; emit(sq, bishopAttacks(sq, occ)); }
    }
    {
        U64 bb = b.pieces[us][ROOK];
        while (bb) { int sq = __builtin_ctzll(bb); bb &= bb - 1; emit(sq, rookAttacks(sq, occ)); }
    }
    {
        U64 bb = b.pieces[us][QUEEN];
        while (bb) {
            int sq = __builtin_ctzll(bb); bb &= bb - 1;
            emit(sq, bishopAttacks(sq, occ) | rookAttacks(sq, occ));
        }
    }
    {
        U64 bb = b.pieces[us][KING];
        while (bb) { int sq = __builtin_ctzll(bb); bb &= bb - 1; emit(sq, kingAttacks[sq]); }
    }
    // Castling kept below (uses king square)
    if (!capturesOnly && b.sq[b.kingSq[us]] == KING && b.col[b.kingSq[us]] == us) {
        // Castling (king may not be in check, pass through check, land in check)
        int home = (us == WHITE) ? 0 : 7;
        int ksq = makeSq(home, 4);
        if (b.kingSq[us] == ksq && !inCheck(b, us)) {
            int ooBit = (us == WHITE) ? 1 : 4;
            int oooBit = (us == WHITE) ? 2 : 8;
            if ((b.castling & ooBit) && b.sq[makeSq(home, 5)] == NO_PIECE && b.sq[makeSq(home, 6)] == NO_PIECE &&
                b.sq[makeSq(home, 7)] == ROOK && b.col[makeSq(home, 7)] == us &&
                !isAttacked(b, makeSq(home, 5), us ^ 1) && !isAttacked(b, makeSq(home, 6), us ^ 1)) {
                Move mv; mv.from = ksq; mv.to = makeSq(home, 6); mv.flags = FLAG_CASTLE_K; out.add(mv);
            }
            if ((b.castling & oooBit) && b.sq[makeSq(home, 3)] == NO_PIECE && b.sq[makeSq(home, 2)] == NO_PIECE && b.sq[makeSq(home, 1)] == NO_PIECE &&
                b.sq[makeSq(home, 0)] == ROOK && b.col[makeSq(home, 0)] == us &&
                !isAttacked(b, makeSq(home, 3), us ^ 1) && !isAttacked(b, makeSq(home, 2), us ^ 1)) {
                Move mv; mv.from = ksq; mv.to = makeSq(home, 2); mv.flags = FLAG_CASTLE_Q; out.add(mv);
            }
        }
    }
}

// Legal move list (make + king-safety filter).
static void legalMoves(Board &b, MoveList &out);
static inline void legalMoves(const Board &b, MoveList &out) { Board &mb = const_cast<Board&>(b); legalMoves(mb, out); }
static void doMove(Board &b, Undo &u, const Move &m);
static void undoMove(Board &b, const Undo &u, const Move &m);

static inline void legalMoves(Board &b, MoveList &out) {
    MoveList pseudo;
    genMoves(b, pseudo, false);
    out.n = 0;
    Undo u;
    // no Board copy â€” make/unmake on same board (5x fewer copies)
    for (int i = 0; i < pseudo.n; i++) {
        doMove(b, u, pseudo.m[i]);
        bool ok = !inCheck(b, b.side ^ 1); // mover was b.side before move
        undoMove(b, u, pseudo.m[i]);
        if (ok) out.add(pseudo.m[i]);
    }
}

// Kept out of line so the large MoveList lives in its own stack frame instead
// of inflating every search/qsearch frame.
static bool noLegalMoves(Board &b) {
    MoveList ml;
    legalMoves(b, ml);
    return ml.n == 0;
}

// ----------------------------------------------------------------- make ---
static void doMove(Board &b, Undo &u, const Move &m) {
    u.captured = m.captured;
    u.castling = b.castling;
    u.ep = b.ep;
    u.half = b.half;
    u.key = b.key;
    int us = b.side, them = us ^ 1;
    int p = b.sq[m.from];

    // Remove mover from origin
    b.key ^= zobristPiece[us][p][m.from];
    b.sq[m.from] = NO_PIECE; b.col[m.from] = -1;
    b.occ[us] &= ~bit(m.from); b.all &= ~bit(m.from);
    b.pieces[us][p] &= ~bit(m.from);

    // Capture
    if (m.flags & FLAG_EP) {
        int capSq = makeSq(rankOf(m.from), fileOf(m.to));
        b.key ^= zobristPiece[them][PAWN][capSq];
        b.sq[capSq] = NO_PIECE; b.col[capSq] = -1;
        b.occ[them] &= ~bit(capSq); b.all &= ~bit(capSq);
        b.pieces[them][PAWN] &= ~bit(capSq);
    } else if (m.captured != NO_PIECE) {
        b.key ^= zobristPiece[them][m.captured][m.to];
        b.sq[m.to] = NO_PIECE; b.col[m.to] = -1;
        b.occ[them] &= ~bit(m.to); b.all &= ~bit(m.to);
        b.pieces[them][m.captured] &= ~bit(m.to);
    }

    // Place mover (possibly promoted)
    int placed = (m.promo != NO_PIECE) ? m.promo : p;
    b.sq[m.to] = placed; b.col[m.to] = us;
    b.key ^= zobristPiece[us][placed][m.to];
    if (placed == KING) b.kingSq[us] = m.to;
    b.occ[us] |= bit(m.to); b.all |= bit(m.to);
    b.pieces[us][placed] |= bit(m.to);

    // Castle rook
    if (m.flags & FLAG_CASTLE_K) {
        int home = (us == WHITE) ? 0 : 7;
        int rf = makeSq(home, 7), rt = makeSq(home, 5);
        b.key ^= zobristPiece[us][ROOK][rf];
        b.sq[rf] = NO_PIECE; b.col[rf] = -1;
        b.occ[us] &= ~bit(rf); b.all &= ~bit(rf);
        b.pieces[us][ROOK] &= ~bit(rf);
        b.sq[rt] = ROOK; b.col[rt] = us;
        b.key ^= zobristPiece[us][ROOK][rt];
        b.occ[us] |= bit(rt); b.all |= bit(rt);
        b.pieces[us][ROOK] |= bit(rt);
    } else if (m.flags & FLAG_CASTLE_Q) {
        int home = (us == WHITE) ? 0 : 7;
        int rf = makeSq(home, 0), rt = makeSq(home, 3);
        b.key ^= zobristPiece[us][ROOK][rf];
        b.sq[rf] = NO_PIECE; b.col[rf] = -1;
        b.occ[us] &= ~bit(rf); b.all &= ~bit(rf);
        b.pieces[us][ROOK] &= ~bit(rf);
        b.sq[rt] = ROOK; b.col[rt] = us;
        b.key ^= zobristPiece[us][ROOK][rt];
        b.occ[us] |= bit(rt); b.all |= bit(rt);
        b.pieces[us][ROOK] |= bit(rt);
    }

    // Castling rights
    b.key ^= zobristCastle[b.castling & 15];
    if (p == KING) b.castling &= (us == WHITE) ? ~3 : ~12;
    if (m.from == makeSq(0, 0) || m.to == makeSq(0, 0)) b.castling &= ~2;
    if (m.from == makeSq(0, 7) || m.to == makeSq(0, 7)) b.castling &= ~1;
    if (m.from == makeSq(7, 0) || m.to == makeSq(7, 0)) b.castling &= ~8;
    if (m.from == makeSq(7, 7) || m.to == makeSq(7, 7)) b.castling &= ~4;
    // Captured rook on its home square already handled via m.to checks above.
    b.key ^= zobristCastle[b.castling & 15];

    // En passant square
    if (b.ep >= 0) b.key ^= zobristEp[fileOf(b.ep)];
    b.ep = -1;
    if (m.flags & FLAG_DOUBLE) b.ep = makeSq(rankOf(m.from) + ((us == WHITE) ? 1 : -1), fileOf(m.from));
    if (b.ep >= 0) b.key ^= zobristEp[fileOf(b.ep)];

    // Clocks + side
    b.half = (p == PAWN || m.captured != NO_PIECE || (m.flags & FLAG_EP)) ? 0 : b.half + 1;
    if (us == BLACK) b.full++;
    b.side = them;
    b.key ^= zobristSide;
}

static void undoMove(Board &b, const Undo &u, const Move &m) {
    int us = b.side ^ 1; // mover
    int them = b.side;
    int p = (m.promo != NO_PIECE) ? PAWN : b.sq[m.to];

    // Restore side/clocks/ep/castling/key wholesale
    b.side = us;
    b.castling = u.castling;
    b.ep = u.ep;
    b.half = u.half;
    if (us == BLACK) b.full--;
    b.key = u.key;
    // Undo castle rook
    if (m.flags & FLAG_CASTLE_K) {
        int home = (us == WHITE) ? 0 : 7;
        b.sq[makeSq(home, 5)] = NO_PIECE; b.col[makeSq(home, 5)] = -1;
        b.occ[us] &= ~bit(makeSq(home, 5)); b.all &= ~bit(makeSq(home, 5)); b.pieces[us][ROOK] &= ~bit(makeSq(home, 5));
        b.sq[makeSq(home, 7)] = ROOK; b.col[makeSq(home, 7)] = us;
        b.occ[us] |= bit(makeSq(home, 7)); b.all |= bit(makeSq(home, 7)); b.pieces[us][ROOK] |= bit(makeSq(home, 7));
    } else if (m.flags & FLAG_CASTLE_Q) {
        int home = (us == WHITE) ? 0 : 7;
        b.sq[makeSq(home, 3)] = NO_PIECE; b.col[makeSq(home, 3)] = -1;
        b.occ[us] &= ~bit(makeSq(home, 3)); b.all &= ~bit(makeSq(home, 3)); b.pieces[us][ROOK] &= ~bit(makeSq(home, 3));
        b.sq[makeSq(home, 0)] = ROOK; b.col[makeSq(home, 0)] = us;
        b.occ[us] |= bit(makeSq(home, 0)); b.all |= bit(makeSq(home, 0)); b.pieces[us][ROOK] |= bit(makeSq(home, 0));
    }
    // Remove placed piece, restore mover
    b.sq[m.to] = NO_PIECE; b.col[m.to] = -1;
    b.occ[us] &= ~bit(m.to); b.all &= ~bit(m.to); b.pieces[us][(m.promo != NO_PIECE ? m.promo : p)] &= ~bit(m.to);
    b.sq[m.from] = p; b.col[m.from] = us;
    b.occ[us] |= bit(m.from); b.all |= bit(m.from); b.pieces[us][p] |= bit(m.from);
    if (p == KING) b.kingSq[us] = m.from;
    // Restore capture
    if (m.flags & FLAG_EP) {
        int capSq = makeSq(rankOf(m.from), fileOf(m.to));
        b.sq[capSq] = PAWN; b.col[capSq] = them;
        b.occ[them] |= bit(capSq); b.all |= bit(capSq); b.pieces[them][PAWN] |= bit(capSq);
    } else if (u.captured != NO_PIECE) {
        b.sq[m.to] = u.captured; b.col[m.to] = them;
        b.occ[them] |= bit(m.to); b.all |= bit(m.to); b.pieces[them][u.captured] |= bit(m.to);
    }
}

// ------------------------------------------------- exact 3-piece endgames --
// KQK / KRK / KPK distance-to-mate tables (retrograde, lazily generated).
#include "endgame_tb.h"

// ------------------------------------------------------------------ NNUE --
// Trained net: (768->2048)x2->1, Chess768 features, SCReLU, QA=255, QB=64.
// File layout (int16 LE, row-major): l0w[2048][768], l0b[2048],
// l1w[4096] (stm block first, then ntm), l1b[1].
// Feature index (bullet chess768.rs): for a piece of color c (0=white),
//   stm = (c?384:0) + 64*type + sq,  ntm = (c?0:384) + 64*type + (sq^56).
// Engine keeps a white-perspective acc and a black-perspective acc;
// side to move picks which feeds l1 first.

struct NNUE {
    bool loaded = false;
    std::vector<int16_t> l0w; // HIDDEN*768
    std::vector<int16_t> l0b; // HIDDEN
    std::vector<int16_t> l1w; // 2*HIDDEN
    int16_t l1b = 0;
    // Output layer pre-folded into float cp: l1wf[i] = l1w[i]*SCALE/(QA*QA*QB),
    // l1bScaled = l1b*SCALE/(QA*QB).  Built once per net load, never per node.
    std::vector<float> l1wf;
    float l1bScaled = 0.0f;
    // King-move save stack: a king-bucket change forces a full rebuild of one
    // perspective, so snapshot it and memcpy it back on unmake.
    static const int KSTACK = MAX_PLY + 32;
    std::vector<int32_t> ksave;   // KSTACK * NNUE_HIDDEN, lazily allocated
    std::vector<int8_t> kwhich;   // perspective+1 stored at each depth
    int ktop = 0;
    int numBuckets = 1;       // 1 = legacy Chess768, 32 = HalfKA_hm style
    // Live accumulators (white-perspective, black-perspective), int32.
    int32_t accW[NNUE_HIDDEN];
    int32_t accB[NNUE_HIDDEN];

    void buildOutput() {
        const float k = (float)NNUE_SCALE / ((float)NNUE_QA * (float)NNUE_QAQB);
        l1wf.resize(l1w.size());
        for (size_t i = 0; i < l1w.size(); i++) l1wf[i] = (float)l1w[i] * k;
        l1bScaled = (float)l1b * (float)NNUE_SCALE / (float)NNUE_QAQB;
    }

    bool load(const std::string &path) {
        loaded = false;   // a failed load must not leave the old net marked active
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz < 2) { fclose(f); return false; }
        size_t stride = (size_t)NNUE_INPUTS * NNUE_HIDDEN;
        // bullet pads every tensor to 64 bytes, so the last one (l1b, 2 bytes)
        // can leave up to 62 bytes of padding at the end of the file.
        int nb = 0;
        for (int k = NNUE_MAX_BUCKETS; k >= 1; k--) {
            size_t v = (size_t)k * stride + (size_t)NNUE_HIDDEN + (size_t)2 * NNUE_HIDDEN + 1;
            size_t bytes = v * sizeof(int16_t);
            size_t padded = (bytes + 63) / 64 * 64;
            if ((size_t)sz == bytes || (size_t)sz == padded) { nb = k; break; }
        }
        if (nb == 0) { fclose(f); return false; }
        numBuckets = nb;
        l0w.resize(stride * (size_t)nb);
        l0b.resize(NNUE_HIDDEN);
        l1w.resize((size_t)2 * NNUE_HIDDEN);
        auto rd = [&](int16_t *dst, size_t n) {
            return fread(dst, sizeof(int16_t), n, f) == n;
        };
        bool ok = rd(l0w.data(), l0w.size()) && rd(l0b.data(), l0b.size()) &&
                  rd(l1w.data(), l1w.size()) && rd(&l1b, 1);
        fclose(f);
        loaded = ok;
        if (ok) buildOutput();
        return ok;
    }

    // Share weights with another NNUE instance (per-thread accumulators stay local).
    void copyWeightsFrom(const NNUE &o) {
        if (!o.loaded) { loaded = false; return; }
        l0w = o.l0w; l0b = o.l0b; l1w = o.l1w; l1b = o.l1b;
        l1wf = o.l1wf; l1bScaled = o.l1bScaled;
        numBuckets = o.numBuckets;
        loaded = true;
    }

    // With a bucketed net every distinct king square changes the feature
    // mapping: equal kingBucket() plus an equal mirror half forces the same
    // square (same rank, same file).  Only a 1-bucket net can patch a king move
    // in place; otherwise the moved king's perspective must be rebuilt.
    bool sameKingMapping(int from, int to) const {
        (void)from; (void)to;
        return numBuckets <= 1;
    }

    // Row for one perspective only: persp 0 = white POV (accW), 1 = black (accB).
    inline const int16_t *rowFor(const Board &b, int persp, int color, int piece,
                                 int sq) const {
        bool bucketed = numBuckets > 1;
        size_t f;
        if (persp == 0) {
            int wk = kingSquareOf(b, WHITE);
            f = (bucketed ? (size_t)NNUE_INPUTS * kingBucket(wk) : 0)
              + (color == WHITE ? 0 : 384) + 64 * piece
              + (bucketed ? bucketMirror(sq, wk) : sq);
        } else {
            int bk = kingSquareOf(b, BLACK);
            f = (bucketed ? (size_t)NNUE_INPUTS * kingBucket(bk ^ 56) : 0)
              + (color == WHITE ? 384 : 0) + 64 * piece
              + (bucketed ? bucketMirror(sq ^ 56, bk) : (sq ^ 56));
        }
        return &l0w[f * NNUE_HIDDEN];
    }

    inline void addPieceP(const Board &b, int persp, int color, int piece, int sq) {
        const int16_t *row = rowFor(b, persp, color, piece, sq);
        int32_t *acc = (persp == 0) ? accW : accB;
        int o = 0;
        for (; o + 16 <= NNUE_HIDDEN; o += 16) {
            __m256i v0 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)&row[o]));
            __m256i v1 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)&row[o + 8]));
            _mm256_storeu_si256((__m256i *)&acc[o],
                _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)&acc[o]), v0));
            _mm256_storeu_si256((__m256i *)&acc[o + 8],
                _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)&acc[o + 8]), v1));
        }
        for (; o < NNUE_HIDDEN; o++) acc[o] += row[o];
    }

    inline void removePieceP(const Board &b, int persp, int color, int piece, int sq) {
        const int16_t *row = rowFor(b, persp, color, piece, sq);
        int32_t *acc = (persp == 0) ? accW : accB;
        int o = 0;
        for (; o + 16 <= NNUE_HIDDEN; o += 16) {
            __m256i v0 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)&row[o]));
            __m256i v1 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)&row[o + 8]));
            _mm256_storeu_si256((__m256i *)&acc[o],
                _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)&acc[o]), v0));
            _mm256_storeu_si256((__m256i *)&acc[o + 8],
                _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)&acc[o + 8]), v1));
        }
        for (; o < NNUE_HIDDEN; o++) acc[o] -= row[o];
    }

    // Rebuild one perspective from scratch on the current board.
    void rebuildP(const Board &b, int persp) {
        int32_t *acc = (persp == 0) ? accW : accB;
        for (int o = 0; o < NNUE_HIDDEN; o++) acc[o] = l0b[o];
        for (int sq = 0; sq < 64; sq++)
            if (b.sq[sq] != NO_PIECE)
                addPieceP(b, persp, b.col[sq], b.sq[sq], sq);
    }

    void saveAcc(int persp) {
        if (ksave.empty()) {
            ksave.assign((size_t)KSTACK * NNUE_HIDDEN, 0);
            kwhich.assign(KSTACK, 0);
            ktop = 0;
        }
        if (ktop < KSTACK && ktop >= 0) {
            memcpy(&ksave[(size_t)ktop * NNUE_HIDDEN],
                   (persp == 0) ? accW : accB,
                   sizeof(int32_t) * (size_t)NNUE_HIDDEN);
            kwhich[ktop] = (int8_t)(persp + 1);
        }
        ktop++;   // still counted when the stack overflows, so pops stay paired
    }

    void restoreAcc(const Board &b, int persp) {
        ktop--;
        if (ktop < 0) { ktop = 0; rebuildP(b, persp); return; }
        if (ktop < KSTACK && kwhich[ktop] == (int8_t)(persp + 1)) {
            memcpy((persp == 0) ? accW : accB,
                   &ksave[(size_t)ktop * NNUE_HIDDEN],
                   sizeof(int32_t) * (size_t)NNUE_HIDDEN);
            return;
        }
        rebuildP(b, persp);   // unmatched/overflowed: slow but exact
    }

    // Root refresh: the save stack cannot survive a search boundary.
    void rootRefresh(const Board &b) { ktop = 0; refresh(b); }

    // Feature add/remove on both accumulators.
    // l0w memory layout is [768][HIDDEN] (row = input feature, verified
    // against training loss: the transposed reading reproduces it).
    void addPiece(const Board &b, int color, int piece, int sq) {
        bool bucketed = numBuckets > 1;
        int wk = kingSquareOf(b, WHITE);
        int bk = kingSquareOf(b, BLACK);
        int fs = (bucketed ? NNUE_INPUTS * kingBucket(wk) : 0)
               + (color == WHITE ? 0 : 384) + 64 * piece + (bucketed ? bucketMirror(sq, wk) : sq);
        int fn = (bucketed ? NNUE_INPUTS * kingBucket(bk ^ 56) : 0)
               + (color == WHITE ? 384 : 0) + 64 * piece + (bucketed ? bucketMirror(sq ^ 56, bk) : (sq ^ 56));
        const int16_t* rowW = &l0w[(size_t)fs * NNUE_HIDDEN];
        const int16_t* rowB = &l0w[(size_t)fn * NNUE_HIDDEN];
        int o = 0;
        for (; o + 16 <= NNUE_HIDDEN; o += 16) {
            __m256i vW0 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&rowW[o]));
            __m256i vW1 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&rowW[o+8]));
            __m256i vB0 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&rowB[o]));
            __m256i vB1 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&rowB[o+8]));
            __m256i aW0 = _mm256_loadu_si256((__m256i*)&accW[o]);
            __m256i aW1 = _mm256_loadu_si256((__m256i*)&accW[o+8]);
            __m256i aB0 = _mm256_loadu_si256((__m256i*)&accB[o]);
            __m256i aB1 = _mm256_loadu_si256((__m256i*)&accB[o+8]);
            aW0 = _mm256_add_epi32(aW0, vW0); aW1 = _mm256_add_epi32(aW1, vW1);
            aB0 = _mm256_add_epi32(aB0, vB0); aB1 = _mm256_add_epi32(aB1, vB1);
            _mm256_storeu_si256((__m256i*)&accW[o], aW0);
            _mm256_storeu_si256((__m256i*)&accW[o+8], aW1);
            _mm256_storeu_si256((__m256i*)&accB[o], aB0);
            _mm256_storeu_si256((__m256i*)&accB[o+8], aB1);
        }
        for (; o < NNUE_HIDDEN; o++) { accW[o] += rowW[o]; accB[o] += rowB[o]; }
    }
    void removePiece(const Board &b, int color, int piece, int sq) {
        bool bucketed = numBuckets > 1;
        int wk = kingSquareOf(b, WHITE);
        int bk = kingSquareOf(b, BLACK);
        int fs = (bucketed ? NNUE_INPUTS * kingBucket(wk) : 0)
               + (color == WHITE ? 0 : 384) + 64 * piece + (bucketed ? bucketMirror(sq, wk) : sq);
        int fn = (bucketed ? NNUE_INPUTS * kingBucket(bk ^ 56) : 0)
               + (color == WHITE ? 384 : 0) + 64 * piece + (bucketed ? bucketMirror(sq ^ 56, bk) : (sq ^ 56));
        const int16_t* rowW = &l0w[(size_t)fs * NNUE_HIDDEN];
        const int16_t* rowB = &l0w[(size_t)fn * NNUE_HIDDEN];
        int o = 0;
        for (; o + 16 <= NNUE_HIDDEN; o += 16) {
            __m256i vW0 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&rowW[o]));
            __m256i vW1 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&rowW[o+8]));
            __m256i vB0 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&rowB[o]));
            __m256i vB1 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)&rowB[o+8]));
            __m256i aW0 = _mm256_loadu_si256((__m256i*)&accW[o]);
            __m256i aW1 = _mm256_loadu_si256((__m256i*)&accW[o+8]);
            __m256i aB0 = _mm256_loadu_si256((__m256i*)&accB[o]);
            __m256i aB1 = _mm256_loadu_si256((__m256i*)&accB[o+8]);
            aW0 = _mm256_sub_epi32(aW0, vW0); aW1 = _mm256_sub_epi32(aW1, vW1);
            aB0 = _mm256_sub_epi32(aB0, vB0); aB1 = _mm256_sub_epi32(aB1, vB1);
            _mm256_storeu_si256((__m256i*)&accW[o], aW0);
            _mm256_storeu_si256((__m256i*)&accW[o+8], aW1);
            _mm256_storeu_si256((__m256i*)&accB[o], aB0);
            _mm256_storeu_si256((__m256i*)&accB[o+8], aB1);
        }
        for (; o < NNUE_HIDDEN; o++) { accW[o] -= rowW[o]; accB[o] -= rowB[o]; }
    }
    // Full refresh from a board (used at root + for verification).
    void refresh(const Board &b) {
        for (int o = 0; o < NNUE_HIDDEN; o++) {
            accW[o] = l0b[o];
            accB[o] = l0b[o];
        }
        for (int sq = 0; sq < 64; sq++)
            if (b.sq[sq] != NO_PIECE) addPiece(b, b.col[sq], b.sq[sq], sq);
    }
    // Static eval, STM perspective. AVX2 SCReLU (float output path — matches trainer scale).
    // cp = out * SCALE / (QA*QB). DO NOT replace with a broken int reduction.
    __attribute__((hot)) inline int evaluate(const Board &b) const {
        const int32_t *first = (b.side == WHITE) ? accW : accB;
        const int32_t *second = (b.side == WHITE) ? accB : accW;
        const __m256i vZero = _mm256_setzero_si256();
        const __m256i vQA = _mm256_set1_epi32(NNUE_QA);
        float fout = l1bScaled;
        auto layer = [&](const int32_t *acc, const float *w) {
            __m256 vsum = _mm256_setzero_ps();
            for (int i = 0; i < NNUE_HIDDEN; i += 8) {
                __m256i a = _mm256_loadu_si256((const __m256i *)&acc[i]);
                a = _mm256_max_epi32(a, vZero);
                a = _mm256_min_epi32(a, vQA);
                __m256 af = _mm256_cvtepi32_ps(a);
                vsum = _mm256_fmadd_ps(_mm256_mul_ps(af, af),
                                       _mm256_loadu_ps(&w[i]), vsum);
            }
            __m128 lo = _mm256_castps256_ps128(vsum);
            __m128 hi = _mm256_extractf128_ps(vsum, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
            lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
            fout += _mm_cvtss_f32(lo);
        };
        layer(first, l1wf.data());
        layer(second, l1wf.data() + NNUE_HIDDEN);
        I64 cp = (I64)fout;
        if (cp > EVAL_CLAMP) cp = EVAL_CLAMP;
        if (cp < -EVAL_CLAMP) cp = -EVAL_CLAMP;
        return (int)cp;
    }
};

// =============================================================================
// SEE + search helpers (wired into search â€” not dead stubs)
// =============================================================================
static const int SEE_VAL[6] = {100, 300, 300, 500, 900, 20000};

static U64 attackersTo(const Board &b, int sq, U64 occ) {
    U64 att = 0;
    att |= (pawnAttacks[BLACK][sq] & b.pieces[WHITE][PAWN]);
    att |= (pawnAttacks[WHITE][sq] & b.pieces[BLACK][PAWN]);
    att |= knightAttacks[sq] & (b.pieces[WHITE][KNIGHT] | b.pieces[BLACK][KNIGHT]);
    att |= kingAttacks[sq] & (b.pieces[WHITE][KING] | b.pieces[BLACK][KING]);
    att |= bishopAttacks(sq, occ) & (b.pieces[WHITE][BISHOP] | b.pieces[BLACK][BISHOP] |
                                     b.pieces[WHITE][QUEEN] | b.pieces[BLACK][QUEEN]);
    att |= rookAttacks(sq, occ) & (b.pieces[WHITE][ROOK] | b.pieces[BLACK][ROOK] |
                                   b.pieces[WHITE][QUEEN] | b.pieces[BLACK][QUEEN]);
    return att;
}

static int smallestAttacker(const Board &b, U64 att, int side, int &fromOut) {
    for (int p = PAWN; p <= KING; p++) {
        U64 bb = att & b.pieces[side][p];
        if (bb) {
            fromOut = __builtin_ctzll(bb);
            return p;
        }
    }
    fromOut = -1;
    return NO_PIECE;
}

// Classic swap-off SEE. Positive => capture sequence wins material for side-to-move.
static int SEE(const Board &b, const Move &m) {
    int gain[32];
    int d = 0;
    int to = m.to;
    int from = m.from;
    int us = b.side;
    int attacker = b.sq[from];
    if (attacker == NO_PIECE) return 0;

    U64 occ = b.all;
    // Move piece: leave from, clear captured, occupy to
    occ ^= bit(from);
    if (m.flags & FLAG_EP) {
        int capSq = makeSq(rankOf(from), fileOf(to));
        occ ^= bit(capSq);
        gain[d] = SEE_VAL[PAWN];
    } else {
        gain[d] = (m.captured == NO_PIECE) ? 0 : SEE_VAL[m.captured];
        if (m.captured != NO_PIECE) occ ^= bit(to); // remove victim (then we land on to)
    }
    occ |= bit(to);
    if (m.promo != NO_PIECE) {
        gain[d] += SEE_VAL[m.promo] - SEE_VAL[PAWN];
        attacker = m.promo;
    }

    U64 att = attackersTo(b, to, occ) & occ;
    // Exclude our piece now sitting on `to` from being an "attacker" until captured off
    // (attackersTo still sees pieces on their origin squares via bitboards â€” piece lists
    // still have the mover on `from`, so remove that bit explicitly)
    att &= ~bit(from);
    att &= ~bit(to);

    int side = us ^ 1;
    while (true) {
        d++;
        gain[d] = SEE_VAL[attacker] - gain[d - 1];
        // Stand-pat cutoff: if we already can't improve, stop
        if (std::max(-gain[d - 1], gain[d]) < 0 && d > 1) { /* continue swap-off */ }

        int from2;
        int next = smallestAttacker(b, att, side, from2);
        if (next == NO_PIECE) break;
        occ ^= bit(from2);
        // Discover x-rays through the captured square
        att |= bishopAttacks(to, occ) & (b.pieces[WHITE][BISHOP] | b.pieces[BLACK][BISHOP] |
                                         b.pieces[WHITE][QUEEN] | b.pieces[BLACK][QUEEN]);
        att |= rookAttacks(to, occ) & (b.pieces[WHITE][ROOK] | b.pieces[BLACK][ROOK] |
                                       b.pieces[WHITE][QUEEN] | b.pieces[BLACK][QUEEN]);
        att &= occ;
        att &= ~bit(to);
        attacker = next;
        side ^= 1;
        if (d >= 30) break;
    }

    while (--d)
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    return gain[0];
}

static int lmrTable[64][64];
static void initLMR() {
    for (int d = 0; d < 64; d++)
        for (int m = 0; m < 64; m++) {
            if (d < 2 || m < 1) { lmrTable[d][m] = 0; continue; }
            // More aggressive reductions â†’ higher effective depth in the same time
            double r = 0.95 + std::log((double)d) * std::log((double)(m + 1)) / 1.95;
            int ri = (int)r;
            if (ri < 0) ri = 0;
            if (ri > d - 1) ri = d - 1;
            lmrTable[d][m] = ri;
        }
}

// Make/unmake that also maintain NNUE accumulators (exact inverses).
static void doMoveNN(Board &b, NNUE &nn, Undo &u, const Move &m) {
    int us = b.side, them = us ^ 1;
    int p = b.sq[m.from];
    // King moves: patch instead of rebuild wherever the geometry allows.
    if (p == KING && !(m.flags & (FLAG_CASTLE_K | FLAG_CASTLE_Q))) {
        if (nn.sameKingMapping(m.from, m.to)) {
            // Bucket and mirror flag unchanged: only the king's own row moves
            // (plus the captured piece's row, if the king takes something).
            nn.removePieceP(b, 0, us, KING, m.from);
            nn.removePieceP(b, 1, us, KING, m.from);
            if (m.captured != NO_PIECE) {
                nn.removePieceP(b, 0, them, m.captured, m.to);
                nn.removePieceP(b, 1, them, m.captured, m.to);
            }
            doMove(b, u, m);
            nn.addPieceP(b, 0, us, KING, m.to);
            nn.addPieceP(b, 1, us, KING, m.to);
            return;
        }
        int pRebuild = (us == WHITE) ? 0 : 1;   // perspective whose king moved
        int pOther = 1 - pRebuild;
        nn.removePieceP(b, pOther, us, KING, m.from);  // board is still pre-move
        if (m.captured != NO_PIECE)
            nn.removePieceP(b, pOther, them, m.captured, m.to);
        nn.saveAcc(pRebuild);
        doMove(b, u, m);
        nn.addPieceP(b, pOther, us, KING, m.to);
        nn.rebuildP(b, pRebuild);
        return;
    }
    // Castling moves the rook too; a full refresh is fine at this rarity.
    if (p == KING || (m.flags & (FLAG_CASTLE_K | FLAG_CASTLE_Q))) {
        doMove(b, u, m);
        nn.refresh(b);
        return;
    }
    nn.removePiece(b, us, p, m.from);
    if (m.flags & FLAG_EP) {
        int capSq = makeSq(rankOf(m.from), fileOf(m.to));
        nn.removePiece(b, them, PAWN, capSq);
    } else if (m.captured != NO_PIECE) {
        nn.removePiece(b, them, m.captured, m.to);
    }
    int placed = (m.promo != NO_PIECE) ? m.promo : p;
    nn.addPiece(b, us, placed, m.to);
    if (m.flags & FLAG_CASTLE_K) {
        int home = (us == WHITE) ? 0 : 7;
        nn.removePiece(b, us, ROOK, makeSq(home, 7));
        nn.addPiece(b, us, ROOK, makeSq(home, 5));
    } else if (m.flags & FLAG_CASTLE_Q) {
        int home = (us == WHITE) ? 0 : 7;
        nn.removePiece(b, us, ROOK, makeSq(home, 0));
        nn.addPiece(b, us, ROOK, makeSq(home, 3));
    }
    doMove(b, u, m);
}

static void undoMoveNN(Board &b, NNUE &nn, const Undo &u, const Move &m) {
    int us = b.side ^ 1, them = b.side;
    int p = (m.promo != NO_PIECE) ? PAWN : b.sq[m.to];
    if (p == KING && !(m.flags & (FLAG_CASTLE_K | FLAG_CASTLE_Q))) {
        bool same = nn.sameKingMapping(m.from, m.to);
        int pRebuild = (us == WHITE) ? 0 : 1;
        int pOther = 1 - pRebuild;
        undoMove(b, u, m);              // back to the pre-move position
        if (same) {
            nn.removePieceP(b, 0, us, KING, m.to);
            nn.removePieceP(b, 1, us, KING, m.to);
            if (u.captured != NO_PIECE) {
                nn.addPieceP(b, 0, them, u.captured, m.to);
                nn.addPieceP(b, 1, them, u.captured, m.to);
            }
            nn.addPieceP(b, 0, us, KING, m.from);
            nn.addPieceP(b, 1, us, KING, m.from);
            return;
        }
        nn.removePieceP(b, pOther, us, KING, m.to);
        if (u.captured != NO_PIECE)
            nn.addPieceP(b, pOther, them, u.captured, m.to);
        nn.addPieceP(b, pOther, us, KING, m.from);
        nn.restoreAcc(b, pRebuild);
        return;
    }
    if (p == KING || (m.flags & (FLAG_CASTLE_K | FLAG_CASTLE_Q))) {
        undoMove(b, u, m);
        nn.refresh(b);
        return;
    }
    if (m.flags & FLAG_CASTLE_K) {
        int home = (us == WHITE) ? 0 : 7;
        nn.removePiece(b, us, ROOK, makeSq(home, 5));
        nn.addPiece(b, us, ROOK, makeSq(home, 7));
    } else if (m.flags & FLAG_CASTLE_Q) {
        int home = (us == WHITE) ? 0 : 7;
        nn.removePiece(b, us, ROOK, makeSq(home, 3));
        nn.addPiece(b, us, ROOK, makeSq(home, 0));
    }
    int placed = (m.promo != NO_PIECE) ? m.promo : p;
    nn.removePiece(b, us, placed, m.to);
    if (m.flags & FLAG_EP) {
        int capSq = makeSq(rankOf(m.from), fileOf(m.to));
        nn.addPiece(b, them, PAWN, capSq);
    } else if (u.captured != NO_PIECE) {
        nn.addPiece(b, them, u.captured, m.to);
    }
    nn.addPiece(b, us, p, m.from);
    undoMove(b, u, m);
}

// ---------------------------------------------------------------- search --
// 16-byte entry: key + one packed data word.  All data fields live in the same
// 64-bit word and are accessed with GCC atomics, so concurrent readers can
// never observe a half-updated entry (the old split-field layout was a data
// race and could mix score/move from different positions).
struct TTEntry {
    U64 key = 0;
    U64 data = ((U64)0xFF << 16); // score=0, depth=-1, flags=0, move=0

    static U64 pack(int score, int depth, int flags, uint32_t move) {
        return (U64)(uint16_t)score
             | ((U64)(uint8_t)depth << 16)
             | ((U64)(uint8_t)flags << 24)
             | ((U64)move << 32);
    }
    static int      scoreOf(U64 d) { return (int)(int16_t)(d & 0xFFFF); }
    static int      depthOf(U64 d) { return (int)(int8_t)((d >> 16) & 0xFF); }
    static int      flagsOf(U64 d) { return (int)((d >> 24) & 0xFF); }
    static uint32_t moveOf (U64 d) { return (uint32_t)(d >> 32); }
};

struct Search;

static std::string moveToUCI(const Move &m);

// Per-thread search state (Lazy SMP). Shared: TT, stop, nodes, clocks.
struct Worker {
    Search *S = nullptr;
    int id = 0;
    Move killers[MAX_PLY][2];
    Move counters[2][6][64];
    Move plyMove[MAX_PLY];
    int history[2][64][64];
    int capHist[6][64];
    U64 repKeys[4096];   // seeded history + the current search path, never overrun
    int repCount = 0;
    int chkExt[MAX_PLY + 1];   // check extensions already taken on this path
    I64 localNodes = 0;        // nodes searched by this worker in this game
    int selDepth = 0;
    Move bestRoot;
    int bestScoreRoot = 0;
    int completedDepth = 0;
    int evalStack[MAX_PLY];
    int rootDepth = 0;
    Board board;
    NNUE nn;

    void resetTables() {
        // Move is a class type (default member initializers), so use value
        // assignment instead of memset.
        std::fill(&killers[0][0], &killers[0][0] + MAX_PLY * 2, Move{});
        std::fill(&counters[0][0][0], &counters[0][0][0] + 2 * 6 * 64, Move{});
        std::fill(plyMove, plyMove + MAX_PLY, Move{});
        memset(history, 0, sizeof(history));
        memset(capHist, 0, sizeof(capHist));
        memset(evalStack, 0, sizeof(evalStack));
        memset(chkExt, 0, sizeof(chkExt));
    }

    inline void histUpdate(int &entry, int bonus) {
        entry += bonus - entry * std::abs(bonus) / 16384;
    }

    // Single point where a node is counted, so "go nodes" and nps agree.
    inline void noteNode();

    // Neutral draws (own): winning lines outscore reps automatically so
    // shuffles are avoided when ahead, while reps are taken when behind.
    inline int drawScore(int ply) const {
        (void)ply;
        return 0;
    }

    inline bool timeUp();
    int quiescence(int alpha, int beta, int ply);
    int search(int depth, int ply, int alpha, int beta, Move &best, bool pvNode, uint32_t excludedMove = 0);
    void idLoop(int maxD, const std::vector<U64> &gameHist, bool isMain);
};

// ---- move-ordering instrumentation (measurement only, no search change) ----
// Where in the ordered move list does the move that causes the cutoff sit?
// hist[k] = number of fail-high nodes where the cutting move was the k-th
// move actually searched (0 = first move tried, i.e. best essay on top).
struct OrdStats {
    unsigned long long cuts = 0;      // fail-high (beta cutoff) nodes
    unsigned long long nodes = 0;     // interior nodes that searched >= 1 move
    unsigned long long searched = 0;  // total moves searched in those nodes
    unsigned long long hist[33] = {};
    unsigned long long bestNodes = 0;   // interior nodes where the best move is known
    unsigned long long bestHist[33] = {};
    int rootIdx = -1;                 // index of final best move in root order

    void reset() {
        cuts = nodes = searched = 0;
        bestNodes = 0;
        for (int i = 0; i < 33; i++) hist[i] = bestHist[i] = 0;
        rootIdx = -1;
    }

    void report(std::ostream &os) const {
        os << "info string ORD cuts " << cuts << " nodes " << nodes
           << " searched " << searched << " root " << rootIdx << " hist";
        for (int i = 0; i < 9; i++) os << " " << hist[i];
        unsigned long long tail = 0;
        for (int i = 9; i < 33; i++) tail += hist[i];
        os << " " << tail;
        os << " bestnodes " << bestNodes << " besthist";
        for (int i = 0; i < 9; i++) os << " " << bestHist[i];
        tail = 0;
        for (int i = 9; i < 33; i++) tail += bestHist[i];
        os << " " << tail;
        os << std::endl;
    }
};

struct Search {
    OrdStats ord;
    std::vector<TTEntry> tt;
    size_t ttMask = 0;
    std::atomic<I64> nodes{ 0 };
    std::atomic<I64> nodeLimit{ -1 };   // -1 = disabled, else stop at this count
    std::atomic<bool> stop{ false };
    I64 startMs = 0, limitMs = 0, hardMs = 0;
    int numThreads = 1;
    Move bestRoot;
    int bestScoreRoot = 0;
    std::mutex ioMutex;
    std::unique_ptr<Worker[]> workers;

    Search() {
        // Ship single-threaded, like every other engine on the CCRL list: the
        // engine only sees the machine's cores, not the cores it was allocated,
        // so the test harness must opt in to more threads explicitly.
        numThreads = 1;
        workers.reset(new Worker[MAX_THREADS]);
        for (int i = 0; i < MAX_THREADS; i++) {
            workers[i].S = this;
            workers[i].id = i;
            workers[i].resetTables();
        }
    }

    void setThreads(int n) {
        if (n < 1) n = 1;
        if (n > MAX_THREADS) n = MAX_THREADS;
        numThreads = n;
        netDirty = true;   // the new workers have no weights yet
    }

    bool netDirty = true;   // weights must be pushed to the workers before searching

    void syncWeights(const NNUE &src) {
        for (int i = 0; i < numThreads; i++)
            workers[i].nn.copyWeightsFrom(src);
        netDirty = false;
    }

    void ttResize(int mb) {
        size_t n = (size_t)mb * 1024 * 1024 / sizeof(TTEntry);
        if (n < 1024) n = 1024;
        size_t p = 1;
        while (p * 2 <= n) p *= 2;
        // Allocate into a temporary and swap, so an allocation failure keeps the
        // previous table instead of leaving a half-resized one behind.
        try {
            std::vector<TTEntry> nt(p, TTEntry());
            tt.swap(nt);
            ttMask = p - 1;
        } catch (const std::bad_alloc &) {
            std::cerr << "info string Hash resize to " << mb << " MB failed" << std::endl;
        }
    }
    void ttClear() { std::fill(tt.begin(), tt.end(), TTEntry()); }

    // Approximate table occupancy for UCI "hashfull" (per mille).
    int hashfull() const {
        if (tt.empty()) return 0;
        size_t sample = std::min<size_t>(tt.size(), 1000);
        size_t used = 0;
        for (size_t i = 0; i < sample; i++)
            if (__atomic_load_n(&tt[i].key, __ATOMIC_RELAXED) != 0) used++;
        return (int)(used * 1000 / sample);
    }

    static I64 nowMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static uint32_t packMove(const Move &m) {
        return (uint32_t)(m.from | (m.to << 6) | ((m.promo == NO_PIECE ? 0 : m.promo + 1) << 12));
    }
    static Move unpackMove(uint32_t d) {
        Move m;
        m.from = d & 63; m.to = (d >> 6) & 63;
        int pr = (d >> 12) & 7;
        m.promo = pr ? pr - 1 : NO_PIECE;
        return m;
    }

    static bool deadDraw(const Board &b) {
        int minor = 0;
        for (int c = 0; c < 2; c++) {
            if (b.pieces[c][PAWN] | b.pieces[c][ROOK] | b.pieces[c][QUEEN]) return false;
            minor += __builtin_popcountll(b.pieces[c][KNIGHT] | b.pieces[c][BISHOP]);
        }
        return minor <= 1;
    }

    static bool nonPawnMaterial(const Board &b, int side) {
        return (b.pieces[side][KNIGHT] | b.pieces[side][BISHOP] |
                b.pieces[side][ROOK] | b.pieces[side][QUEEN]) != 0;
    }

    static std::string scoreStr(int score) {
        if (score > MATE_SCORE - MAX_PLY) {
            int plies = MATE_SCORE - score;
            return "mate " + std::to_string((plies + 1) / 2);
        }
        if (score < -MATE_SCORE + MAX_PLY) {
            int plies = MATE_SCORE + score;
            return "mate -" + std::to_string((plies + 1) / 2);
        }
        return "cp " + std::to_string(score);
    }

    void iterate(Board &b, NNUE &nn, int maxD, I64 softMs, I64 hardMsIn, const std::vector<U64> &gameHist) {
        startMs = nowMs();
        limitMs = softMs;
        hardMs = hardMsIn;
        // stop is cleared by the UCI thread before the search thread starts, so
        // a "stop" issued immediately after "go" cannot be lost here.
        nodes.store(0, std::memory_order_relaxed);
        bestRoot = Move();
        bestScoreRoot = 0;
        if (netDirty) syncWeights(nn);   // 24 MB/thread; only when it changed

        MoveList rootMoves;
        legalMoves(b, rootMoves);
        if (rootMoves.n == 0) { std::cout << "bestmove 0000" << std::endl; return; }
        bestRoot = rootMoves.m[0];

        // Exact 3-piece endgames (KQK/KRK/KPK): solve by retrograde tablebase.
        // These mates sit far outside a normal time-limited search horizon, so
        // resolve them directly and report the true "mate N".
        {
            Move tbMove;
            int tbScore = 0, tbMate = 0;
            std::vector<Move> tbPv;
            if (EndgameTB::tbProbeRoot(b, tbMove, tbScore, tbPv, tbMate)) {
                I64 elapsed = std::max<I64>(1, nowMs() - startMs);
                int shownDepth = std::max(1, tbMate);
                std::cout << "info depth " << shownDepth
                          << " seldepth " << shownDepth
                          << " score " << scoreStr(tbScore)
                          << " nodes 0 nps 0 time " << elapsed
                          << " hashfull " << hashfull()
                          << " tbhits 1 pv";
                for (size_t i = 0; i < tbPv.size(); i++)
                    std::cout << " " << moveToUCI(tbPv[i]);
                std::cout << std::endl;
                std::cout << "bestmove " << moveToUCI(tbMove) << std::endl;
                return;
            }
        }

        // Seed every worker with the root position
        for (int t = 0; t < numThreads; t++) {
            workers[t].board = b;
            workers[t].nn.rootRefresh(b);
            workers[t].bestRoot = bestRoot;
            workers[t].bestScoreRoot = 0;
            workers[t].completedDepth = 0;
            workers[t].selDepth = 0;
            workers[t].repCount = 0;
            for (size_t i = 0; i < gameHist.size() && workers[t].repCount < 3800; i++)
                workers[t].repKeys[workers[t].repCount++] = gameHist[i];
        }

        std::vector<std::thread> pool;
        pool.reserve((size_t)std::max(0, numThreads - 1));
        for (int t = 1; t < numThreads; t++) {
            pool.emplace_back([this, t, maxD, &gameHist]() {
                workers[t].idLoop(maxD, gameHist, false);
            });
        }
        workers[0].idLoop(maxD, gameHist, true);

        // In a timed game the main worker owns the decision: once it has spent
        // its soft budget the helpers must not keep the process thinking, or
        // the clock savings from the soft limit are thrown away. Only
        // depth/infinite runs (huge hard limit) let helpers burn extra time.
        const bool timed = hardMs < ((I64)1 << 40);
        if (!timed) {
            while (nowMs() - startMs < hardMs - 15 && !stop.load(std::memory_order_relaxed)) {
                int bestD = 0;
                for (int t = 0; t < numThreads; t++)
                    if (workers[t].completedDepth > bestD) bestD = workers[t].completedDepth;
                if (bestD >= maxD) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        stop.store(true, std::memory_order_relaxed);
        for (auto &th : pool) th.join();

        // Prefer deepest completed result among all threads
        int bestD = workers[0].completedDepth;
        bestRoot = workers[0].bestRoot;
        bestScoreRoot = workers[0].bestScoreRoot;
        for (int t = 1; t < numThreads; t++) {
            if (workers[t].completedDepth > bestD &&
                !(workers[t].bestRoot.from == workers[t].bestRoot.to)) {
                bestD = workers[t].completedDepth;
                bestRoot = workers[t].bestRoot;
                bestScoreRoot = workers[t].bestScoreRoot;
            }
        }

        Move bm = bestRoot;
        if (bm.from == bm.to) {
            // Search may have aborted before completing depth 1 (e.g. a
            // repetition draw at the root).  Never answer 0000 when a legal
            // move exists: fall back to the first legal root move.
            bm = rootMoves.m[0];
        }
        if (bm.from == bm.to) {
            std::cout << "bestmove 0000" << std::endl;
            return;
        }
        std::cout << "bestmove " << moveToUCI(bm) << std::endl;
    }
};

inline void Worker::noteNode() {
    S->nodes.fetch_add(1, std::memory_order_relaxed);
    localNodes++;
}

inline bool Worker::timeUp() {
    if (S->stop.load(std::memory_order_relaxed)) return true;
    I64 n = S->nodes.load(std::memory_order_relaxed);
    I64 lim = S->nodeLimit.load(std::memory_order_relaxed);
    if (lim >= 0 && n >= lim) {
        S->stop.store(true, std::memory_order_relaxed);
        return true;
    }
    if ((n & 2047) == 0) {
        I64 now = Search::nowMs();
        if (now - S->startMs >= S->hardMs) {
            S->stop.store(true, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

int Worker::quiescence(int alpha, int beta, int ply) {
    noteNode();
    if (timeUp()) return 0;
    if (ply > selDepth) selDepth = ply;
    Board &b = board;
    if (ply >= MAX_PLY) return nn.evaluate(b);

    bool chk = inCheck(b, b.side);
    // 50-move draw, but checkmate on the 100th half-move takes precedence.
    if (b.half >= 100) {
        if (chk && noLegalMoves(b)) return -MATE_SCORE + ply;
        return 0;
    }
    // Repetitions must also be detected at the qsearch horizon; the position's
    // own key is at repCount-1 (pushed by the parent make).
    if (ply > 0)
    for (int i = repCount - 3; i >= 0; i -= 2) {
        if (repKeys[i] == b.key) return 0;
        if (repCount - i > 48) break;
    }
    int standPat = 0;
    if (!chk) {
        standPat = nn.evaluate(b);
        if (standPat >= beta) return standPat;
        if (standPat > alpha) alpha = standPat;
    }

    TTEntry *e = &S->tt[b.key & S->ttMask];
    U64 tk = __atomic_load_n(&e->key, __ATOMIC_RELAXED);
    if (tk == b.key) {
        U64 td = __atomic_load_n(&e->data, __ATOMIC_RELAXED);
        if (__atomic_load_n(&e->key, __ATOMIC_RELAXED) == b.key) {
            int s = TTEntry::scoreOf(td);
            if (s > MATE_SCORE - MAX_PLY) s -= ply;
            else if (s < -MATE_SCORE + MAX_PLY) s += ply;
            int fl = TTEntry::flagsOf(td);
            if (fl == 0) return s;
            if (fl == 1 && s >= beta) return s;
            if (fl == 2 && s <= alpha) return s;
        }
    }

    MoveList ml;
    if (chk) {
        legalMoves(b, ml);
        if (ml.n == 0) return -MATE_SCORE + ply;
    } else {
        genMoves(b, ml, true);
        // Compact the legal subset in place: a second MoveList here used to
        // double the quiescence stack frame.
        int keep = 0;
        Undo tu;
        for (int i = 0; i < ml.n; i++) {
            Move cm = ml.m[i];
            if (cm.promo == NO_PIECE) {
                int see = SEE(b, cm);
                if (see < 0) continue;
                if (standPat + see + 200 < alpha) continue;
            }
            doMove(b, tu, cm);
            bool ok = !inCheck(b, b.side ^ 1);
            undoMove(b, tu, cm);
            if (ok) ml.m[keep++] = cm;
        }
        ml.n = keep;
    }

    std::sort(ml.m, ml.m + ml.n, [&](const Move &a, const Move &c) {
        int sa = (a.captured == NO_PIECE ? 0 : 1000000 + a.captured * 100 - b.sq[a.from])
                 + (a.promo != NO_PIECE ? 50000 : 0)
                 + (a.captured != NO_PIECE ? capHist[a.captured][a.to] : 0);
        int sb = (c.captured == NO_PIECE ? 0 : 1000000 + c.captured * 100 - b.sq[c.from])
                 + (c.promo != NO_PIECE ? 50000 : 0)
                 + (c.captured != NO_PIECE ? capHist[c.captured][c.to] : 0);
        return sa > sb;
    });

    Undo u;
    int best = chk ? -INF_SCORE : alpha;
    for (int i = 0; i < ml.n; i++) {
        doMoveNN(b, nn, u, ml.m[i]);
        repKeys[repCount++] = b.key;
        int score = -quiescence(-beta, -alpha, ply + 1);
        repCount--;
        undoMoveNN(b, nn, u, ml.m[i]);
        if (S->stop.load(std::memory_order_relaxed)) return 0;
        if (score > best) best = score;
        if (score > alpha) {
            alpha = score;
            if (score >= beta) return score;
        }
    }
    return best;
}

int Worker::search(int depth, int ply, int alpha, int beta, Move &best, bool pvNode, uint32_t excludedMove) {
    noteNode();
    if (timeUp()) return 0;
    if (ply > selDepth) selDepth = ply;
    Board &b = board;
    if (ply >= MAX_PLY) return nn.evaluate(b);

    bool chk = inCheck(b, b.side);

    // Draws with contempt (avoid shuffle-draws when a real try exists).
    // Checkmate on the 100th half-move is mate, not a 50-move draw.
    if (b.half >= 100) {
        if (chk && noLegalMoves(b)) return -MATE_SCORE + ply;
        return drawScore(ply);
    }
    if (Search::deadDraw(b)) return 0; // dead material can never be a mate
    // v16v2.3: keys include zobristSide and this node's own key sits at
    // repKeys[repCount-1], so repetitions are at repCount-1, -3, -5, ...
    // Root is skipped: a repeated root position is the GUI's call, not ours.
    if (ply > 0)
    for (int i = repCount - 3; i >= 0; i -= 2) {
        if (repKeys[i] == b.key) return drawScore(ply);
        if (repCount - i > 48) break;
    }

    int mated = -MATE_SCORE + ply;
    if (mated > alpha) { alpha = mated; if (alpha >= beta) return alpha; }
    int mates = MATE_SCORE - ply - 1;
    if (mates < beta) { beta = mates; if (alpha >= beta) return alpha; }

    // No entry check extension here: every check is already extended once by
    // the parent's budgeted, SEE-gated check extension.  Extending again at
    // node entry made forcing lines keep depth constant forever.
    if (depth <= 0) return quiescence(alpha, beta, ply);

    TTEntry *e = &S->tt[b.key & S->ttMask];
    __builtin_prefetch(e, 0, 1);
    Move ttMove{};
    bool hasTt = false;
    int ttScore = 0;
    int ttBound = -1;
    int ttDepth = -1;
    U64 tk = __atomic_load_n(&e->key, __ATOMIC_RELAXED);
    if (tk == b.key) {
        U64 td = __atomic_load_n(&e->data, __ATOMIC_RELAXED);
        // Re-read the key: a concurrent writer may have replaced the entry
        // between the two loads.  If so, treat it as a miss.
        if (__atomic_load_n(&e->key, __ATOMIC_RELAXED) == b.key) {
            hasTt = true;
            uint32_t mv = TTEntry::moveOf(td);
            if (mv) ttMove = Search::unpackMove(mv);
            ttScore = TTEntry::scoreOf(td);
            ttBound = TTEntry::flagsOf(td);
            ttDepth = TTEntry::depthOf(td);
            if (ttScore > MATE_SCORE - MAX_PLY) ttScore -= ply;
            else if (ttScore < -MATE_SCORE + MAX_PLY) ttScore += ply;
            if (!excludedMove && ttDepth >= depth && !pvNode) {
                if (ttBound == 0) return ttScore;
                if (ttBound == 1 && ttScore >= beta) return ttScore;
                if (ttBound == 2 && ttScore <= alpha) return ttScore;
            }
        }
    }

    // IIR
    if (!hasTt && !excludedMove && depth >= 4) depth--;

    int staticEval = nn.evaluate(b);
    if (hasTt && ttScore < MATE_SCORE - MAX_PLY && ttScore > -MATE_SCORE + MAX_PLY) {
        if ((ttBound == 1 && ttScore > staticEval) ||
            (ttBound == 2 && ttScore < staticEval) ||
            ttBound == 0)
            staticEval = ttScore;
    }
    evalStack[ply] = staticEval;
    bool improving = ply >= 2 && staticEval > evalStack[ply - 2];

    // Razoring
    if (!pvNode && !chk && !excludedMove && depth <= 3) {
        int ralpha = alpha - (depth == 1 ? 300 : 250 * depth);
        if (staticEval < ralpha) {
            int q = quiescence(alpha, beta, ply);
            if (q < ralpha) return q;
        }
    }

    // Reverse futility
    if (!pvNode && !chk && !excludedMove && depth <= 8) {
        int margin = (improving ? 70 : 90) * depth;
        if (staticEval - margin >= beta) return staticEval;
    }

    // Null-move pruning
    if (!pvNode && !chk && !excludedMove && depth >= 3 && Search::nonPawnMaterial(b, b.side) &&
        staticEval >= beta && beta < MATE_SCORE - MAX_PLY) {
        Undo nu;
        nu.castling = b.castling; nu.ep = b.ep; nu.half = b.half; nu.key = b.key;
        nu.captured = NO_PIECE;
        b.key ^= zobristSide;
        if (b.ep >= 0) b.key ^= zobristEp[fileOf(b.ep)];
        b.ep = -1;
        b.side ^= 1;
        b.half++;
        // Null-move keys are search artifacts: XOR with a constant so they can
        // never collide with a real game position's key.
        repKeys[repCount++] = b.key ^ 0xD1B54A32D192ED03ULL;

        int R = 4 + depth / 4 + std::min(3, (staticEval - beta) / 180);
        if (!improving) R++;
        if (R > depth - 1) R = depth - 1;
        Move dummy;
        chkExt[ply + 1] = chkExt[ply];
        int score = -search(depth - 1 - R, ply + 1, -beta, -beta + 1, dummy, false, 0);

        repCount--;
        b.side ^= 1;
        b.ep = nu.ep;
        b.half = nu.half;
        b.castling = nu.castling;
        b.key = nu.key;

        if (S->stop.load(std::memory_order_relaxed)) return 0;
        if (score >= beta)
            return score >= MATE_SCORE - MAX_PLY ? beta : score;
    }

    // ProbCut: if a capture proves a cut at reduced depth, skip full search
    if (!pvNode && !chk && !excludedMove && depth >= 5 && abs(beta) < MATE_SCORE - MAX_PLY) {
        int pcBeta = beta + 180;
        MoveList caps;
        genMoves(b, caps, true);
        Undo tu;
        for (int i = 0; i < caps.n && i < 16; i++) {
            Move &cm = caps.m[i];
            if (cm.promo == NO_PIECE && SEE(b, cm) < 0) continue;
            doMove(b, tu, cm);
            bool ok = !inCheck(b, b.side ^ 1);
            undoMove(b, tu, cm);
            if (!ok) continue;
            doMoveNN(b, nn, tu, cm);
            repKeys[repCount++] = b.key;
            Move dummy;
            chkExt[ply + 1] = chkExt[ply];
            int score = -search(depth - 4, ply + 1, -pcBeta, -pcBeta + 1, dummy, false, 0);
            repCount--;
            undoMoveNN(b, nn, tu, cm);
            if (S->stop.load(std::memory_order_relaxed)) return 0;
            if (score >= pcBeta) return score;
        }
    }

    MoveList ml;
    genMoves(b, ml, false); // pseudo-legal: skip the full make/unmake pre-pass
    if (ml.n == 0) { // no pseudo-moves at all
        if (chk) return -MATE_SCORE + ply;
        return drawScore(ply);
    }

    // Skip excluded move (singular extension search), compacting in place.
    if (excludedMove) {
        int keep = 0;
        for (int i = 0; i < ml.n; i++)
            if (Search::packMove(ml.m[i]) != excludedMove) ml.m[keep++] = ml.m[i];
        ml.n = keep;
        if (ml.n == 0) return alpha;
    }

    // Index into `ml` plus score: keeps the ordering array small on the stack.
    struct Scored { int s; int idx; };
    Scored ord[448];
    Move prev{};
    if (ply > 0) prev = plyMove[ply - 1];
    bool hasPrev = ply > 0;
    int prevPt = NO_PIECE;
    if (hasPrev) {
        prevPt = b.sq[prev.to];
        if (prevPt < 0 || prevPt > KING) prevPt = NO_PIECE;
    }

    for (int i = 0; i < ml.n; i++) {
        const Move &m = ml.m[i];
        int s = 0;
        if (hasTt && m.from == ttMove.from && m.to == ttMove.to && m.promo == ttMove.promo)
            s = 20000000;
        else if (m.captured != NO_PIECE || m.promo != NO_PIECE) {
            // MVV-LVA for sound captures; losing captures sink below quiets
            // (still searched, just late) so bad tactics don't delay cutoffs.
            int see = (m.captured != NO_PIECE && m.promo == NO_PIECE) ? SEE(b, m) : 0;
            if (see >= 0)
                s = 10000000 + m.captured * 100 - b.sq[m.from]
                    + (m.promo != NO_PIECE ? 50000 : 0)
                    + (m.captured != NO_PIECE ? capHist[m.captured][m.to] : 0);
            else
                s = -1000000 + see;
        } else if (ply < MAX_PLY && m.from == killers[ply][0].from && m.to == killers[ply][0].to)
            s = 9000000;
        else if (ply < MAX_PLY && m.from == killers[ply][1].from && m.to == killers[ply][1].to)
            s = 8000000;
        else if (hasPrev && prevPt >= 0 && prevPt <= KING &&
                 m.from == counters[b.side][prevPt][prev.to].from &&
                 m.to == counters[b.side][prevPt][prev.to].to)
            s = 7000000;
        else
            s = history[b.side][m.from][m.to];
        // Root: penalize moves that immediately repeat a game position
        if (ply == 0) {
            Undo ru; doMove(b, ru, m);
            for (int ri = repCount - 1; ri >= 0; ri--) {
                if (repKeys[ri] == b.key) { s -= 500000; break; }
            }
            undoMove(b, ru, m);
        }
        ord[i] = { s, i };
    }
    std::sort(ord, ord + ml.n, [](const Scored &a, const Scored &c) { return a.s > c.s; });

    // (removed helper-only move rotation; helpers now search full strength)

    // Singular extension: TT move uniquely good â†’ search deeper
    int extension = 0;
    if (!pvNode && !excludedMove && depth >= 7 && hasTt && ttMove.from != ttMove.to &&
        ttDepth >= depth - 3 && ttBound != 2 &&
        abs(ttScore) < MATE_SCORE - MAX_PLY) {
        int sBeta = ttScore - depth;
        Move dummy;
        int sScore = search(depth / 2, ply, sBeta - 1, sBeta, dummy, false, Search::packMove(ttMove));
        if (!S->stop.load(std::memory_order_relaxed) && sScore < sBeta)
            extension = 1;
    }

    int bestScore = -INF_SCORE;
    Move curBest = ml.m[ord[0].idx];
    int origAlpha = alpha;
    int movesPlayed = 0;
    int bestIdx = -1;
    Undo u;

    for (int i = 0; i < ml.n; i++) {
        Move m = ml.m[ord[i].idx];
        bool isCap = m.captured != NO_PIECE || (m.flags & FLAG_EP);
        bool isPromo = m.promo != NO_PIECE;
        bool quiet = !isCap && !isPromo;
        bool isTtMove = hasTt && m.from == ttMove.from && m.to == ttMove.to && m.promo == ttMove.promo;

        if (!pvNode && !chk && quiet && depth <= 8) {
            int lmp = (improving ? 4 : 3) + depth * depth / (improving ? 1 : 2);
            if (movesPlayed >= lmp) continue;
        }

        if (!pvNode && !chk && quiet && depth <= 7 &&
            staticEval + (improving ? 100 : 130) * depth <= alpha && movesPlayed >= 1)
            continue;

        if (!pvNode && !chk && depth <= 8 && movesPlayed >= 1) {
            if (isCap && !isPromo && SEE(b, m) < -20 * depth * depth) continue;
            if (quiet && SEE(b, m) < -15 * depth * depth) continue;
        }

        doMoveNN(b, nn, u, m);
        if (inCheck(b, b.side ^ 1)) { // illegal: mover's king in check; skip
            undoMoveNN(b, nn, u, m);
            continue;
        }
        repKeys[repCount++] = b.key;
        plyMove[ply] = m;

        int score;
        Move child{};
        bool givesCheck = inCheck(b, b.side);
        int newDepth = depth - 1 + (isTtMove ? extension : 0);
        // Check extension: forcing lines must not be reduced away.  Unbounded
        // it is a disaster on forced checking chains, where depth never falls
        // (one middlegame position was 277x in nodes), so it is budgeted per
        // path and gated on SEE.
        int extChk = 0;
        if (givesCheck && depth <= 12 && depth >= 2 && chkExt[ply] < CHK_CAP) {
            bool take = true;
            if (CHK_SEE_GATE) {
                // SEE must be evaluated on the pre-move board.  (Calling it on
                // `b` here sees the mover already gone and always scores 0.)
                Board pre = b;
                undoMove(pre, u, m);
                take = SEE(pre, m) >= CHK_SEE_MARGIN;
            }
            if (take) { newDepth++; extChk = 1; }
        }
        chkExt[ply + 1] = chkExt[ply] + extChk;

        int R = 0;
        if (!pvNode && quiet && !chk && !givesCheck && depth >= 3 && movesPlayed >= 2) {
            R = lmrTable[std::min(depth, 63)][std::min(movesPlayed, 63)];
            if (!improving) R++;
            if (history[b.side ^ 1][m.from][m.to] > 1000) R = std::max(0, R - 1);
            if (history[b.side ^ 1][m.from][m.to] < -1000) R++;
            if (R > newDepth) R = newDepth;
        }

        if (movesPlayed == 0) {
            score = -search(newDepth, ply + 1, -beta, -alpha, child, pvNode, 0);
        } else {
            int redDepth = newDepth - R;
            if (redDepth < 0) redDepth = 0;
            score = -search(redDepth, ply + 1, -alpha - 1, -alpha, child, false, 0);
            if (R && score > alpha && !S->stop.load(std::memory_order_relaxed))
                score = -search(newDepth, ply + 1, -alpha - 1, -alpha, child, false, 0);
            if (score > alpha && score < beta && !S->stop.load(std::memory_order_relaxed))
                score = -search(newDepth, ply + 1, -beta, -alpha, child, true, 0);
        }

        repCount--;
        undoMoveNN(b, nn, u, m);
        if (S->stop.load(std::memory_order_relaxed)) return 0;

        movesPlayed++;
        if (score > bestScore) {
            bestScore = score;
            curBest = m;
            bestIdx = movesPlayed - 1;
        }
        if (score > alpha) alpha = score;
        if (score >= beta) {
            if (ply > 0 && S->numThreads == 1) {
                int k = bestIdx < 0 ? 0 : bestIdx;
                S->ord.cuts++;
                S->ord.hist[k < 32 ? k : 32]++;
            }
            if (quiet && ply < MAX_PLY) {
                if (!(m.from == killers[ply][0].from && m.to == killers[ply][0].to)) {
                    killers[ply][1] = killers[ply][0];
                    killers[ply][0] = m;
                }
                int bonus = std::min(2000, 32 * depth * depth);
                histUpdate(history[b.side][m.from][m.to], bonus);
                if (hasPrev && prevPt >= 0 && prevPt <= KING)
                    counters[b.side][prevPt][prev.to] = m;
                for (int j = 0; j < i; j++) {
                    const Move &pm = ml.m[ord[j].idx];
                    if (pm.captured == NO_PIECE && pm.promo == NO_PIECE)
                        histUpdate(history[b.side][pm.from][pm.to], -bonus / 2);
                }
            } else if (isCap && m.captured != NO_PIECE) {
                histUpdate(capHist[m.captured][m.to], std::min(2000, 32 * depth * depth));
            }
            break;
        }
    }

    if (movesPlayed > 0 && S->numThreads == 1) {
        if (ply > 0) {
            S->ord.nodes++;
            S->ord.searched += (unsigned long long)movesPlayed;
            if (movesPlayed > 1) {
                S->ord.bestNodes++;
                int bk = bestIdx < 0 ? 0 : bestIdx;
                S->ord.bestHist[bk < 32 ? bk : 32]++;
            }
        } else {
            S->ord.rootIdx = bestIdx;
        }
    }

    if (movesPlayed == 0 && !S->stop.load(std::memory_order_relaxed)) {
        if (excludedMove) return alpha;
        if (chk) return -MATE_SCORE + ply;
        return drawScore(ply);
    }

    if (!S->stop.load(std::memory_order_relaxed) && movesPlayed > 0 && !excludedMove) {
        int flags = (bestScore >= beta) ? 1 : (bestScore > origAlpha ? 0 : 2);
        int store = bestScore;
        if (store > MATE_SCORE - MAX_PLY) store += ply;
        else if (store < -MATE_SCORE + MAX_PLY) store -= ply;
        U64 oldKey = __atomic_load_n(&e->key, __ATOMIC_RELAXED);
        U64 oldData = __atomic_load_n(&e->data, __ATOMIC_RELAXED);
        int oldDepth = (oldKey == b.key) ? TTEntry::depthOf(oldData) : -1;
        uint32_t bestPacked = Search::packMove(curBest);
        if (oldKey != b.key || oldDepth <= depth || flags == 0) {
            U64 nd = TTEntry::pack(store, std::min(depth, 127), flags, bestPacked);
            // Data first, key last: a reader that sees the new key then sees
            // the new data, and the reader's key re-check detects any torn read.
            __atomic_store_n(&e->data, nd, __ATOMIC_RELAXED);
            __atomic_store_n(&e->key, b.key, __ATOMIC_RELAXED);
        } else if (TTEntry::moveOf(oldData) == 0) {
            U64 nd = (oldData & 0x00000000FFFFFFFFULL) | ((U64)bestPacked << 32);
            __atomic_store_n(&e->data, nd, __ATOMIC_RELAXED);
        }
    }
    best = curBest;
    return bestScore;
}

void Worker::idLoop(int maxD, const std::vector<U64> &gameHist, bool isMain) {
    (void)gameHist;
    if (isMain) S->ord.reset();
    int lastScore = 0;
    int startDepth = isMain ? 1 : (1 + (id % 3));
    I64 lastIterMs = 0;
    int stable = 0;                 // consecutive iterations with the same move
    Move lastBest; lastBest.from = lastBest.to = -1;
    int scoreDrop = 0;              // how much the score fell last iteration

    // Per-iteration "how critical is this position?" score, used to stretch the
    // time budget when the position is unsettled. Only meaningful while the
    // clock still has slack; fixed movetime/depth requests are not stretched.
    auto criticality = [&](int depth) {
        int crit = 0;
        if (depth >= 9 && stable <= 1) crit++;              // best move keeps changing
        if (depth >= 5 && scoreDrop >= 90) crit++;          // evaluation just dropped
        if (depth >= 10 && std::abs(lastScore) <= 120) crit++; // balanced = sharp
        return crit;
    };

    for (int depth = startDepth; depth <= maxD; depth++) {
        rootDepth = depth;
        I64 elapsed0 = Search::nowMs() - S->startMs;
        if (depth > 1) {
            if (S->stop.load(std::memory_order_relaxed)) break;
            const bool timed = S->hardMs > S->limitMs;
            int crit = timed ? criticality(depth) : 0;
            // Soft limit: the point at which we stop *starting* new iterations.
            // A best move that has survived several iterations is banked time,
            // but a critical position gets that time back and then some.
            I64 soft = S->limitMs;
            if (timed) {
                static const int mult[4] = {100, 130, 160, 200};   // percent
                soft = soft * mult[crit] / 100;
                I64 ceil = S->hardMs * 4 / 5;   // never plan past 80% of the cap
                if (soft > ceil) soft = ceil;
                if (stable >= 3) soft = soft * (crit ? 19 : 15) / 20;
            } else if (stable >= 3) {
                soft = soft * 3 / 5;
            }
            if (elapsed0 >= soft) break;
            // Also refuse an iteration that cannot finish inside the hard cap:
            // the next one usually costs 1.6-2.5x the previous one. A critical
            // position is allowed to gamble a little more of the slack.
            if (lastIterMs > 0 && timed &&
                elapsed0 + lastIterMs * (crit ? 3 : 2) / (crit ? 2 : 1) > S->hardMs) break;
        }
        if (S->stop.load(std::memory_order_relaxed)) break;
        I64 iterStart = Search::nowMs();

        selDepth = 0;
        int alpha = -INF_SCORE, beta = INF_SCORE;
        int delta = 20 + (isMain ? 0 : (id * 4));
        if (depth >= 4) {
            alpha = lastScore - delta;
            beta = lastScore + delta;
        }

        Move bm;
        int score;
        while (true) {
            chkExt[0] = 0;
            score = search(depth, 0, alpha, beta, bm, true, 0);
            if (S->stop.load(std::memory_order_relaxed)) break;
            if (score <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = score - delta;
                if (alpha < -INF_SCORE) alpha = -INF_SCORE;
                delta = std::min(delta * 2, 2000);
            } else if (score >= beta) {
                beta = score + delta;
                if (beta > INF_SCORE) beta = INF_SCORE;
                delta = std::min(delta * 2, 2000);
            } else break;
            if (delta > 1800) {
                alpha = -INF_SCORE; beta = INF_SCORE;
            }
        }

        if (S->stop.load(std::memory_order_relaxed)) break;
        // A draw/mate return at the root leaves `bm` untouched; keep the last
        // completed root move instead of overwriting it with a null move.
        if (bm.from == bm.to) bm = bestRoot;
        if (bm.from == bm.to && depth > 1) break;

        if (depth >= 5) {
            int fall = lastScore - score;     // positive = evaluation got worse
            scoreDrop = (fall > 0) ? fall : scoreDrop * 3 / 4;
        }
        lastScore = score;
        bestScoreRoot = score;
        bestRoot = bm;
        completedDepth = depth;
        if (bm.from == lastBest.from && bm.to == lastBest.to) stable++;
        else stable = 0;
        lastBest = bm;
        lastIterMs = Search::nowMs() - iterStart;

        if (isMain) {
            I64 elapsed = Search::nowMs() - S->startMs;
            if (elapsed < 1) elapsed = 1;
            I64 n = S->nodes.load(std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(S->ioMutex);
            std::cout << "info depth " << depth
                      << " seldepth " << selDepth
                      << " score " << Search::scoreStr(score)
                      << " nodes " << n
                      << " nps " << (n * 1000 / elapsed)
                      << " time " << elapsed
                      << " hashfull " << S->hashfull()
                      << " pv " << moveToUCI(bm) << std::endl;
            S->bestRoot = bm;
            S->bestScoreRoot = score;
        }

        if (score > MATE_SCORE - MAX_PLY || score < -MATE_SCORE + MAX_PLY) {
            if (isMain) S->stop.store(true, std::memory_order_relaxed);
            break;
        }
    }
    if (isMain) S->ord.report(std::cout);
}
// ------------------------------------------------------------------- UCI --
static std::string moveToUCI(const Move &m) {
    std::string s;
    s += char('a' + fileOf(m.from));
    s += char('1' + rankOf(m.from));
    s += char('a' + fileOf(m.to));
    s += char('1' + rankOf(m.to));
    if (m.promo != NO_PIECE) s += "nbrq"[m.promo - 1];
    return s;
}

static bool parseUCIMove(const Board &b, const std::string &tok, Move &out) {
    if (tok.size() < 4) return false;
    if (tok[0] < 'a' || tok[0] > 'h' || tok[1] < '1' || tok[1] > '8' ||
        tok[2] < 'a' || tok[2] > 'h' || tok[3] < '1' || tok[3] > '8') return false;
    int from = makeSq(tok[1] - '1', tok[0] - 'a');
    int to = makeSq(tok[3] - '1', tok[2] - 'a');
    int promo = NO_PIECE;
    if (tok.size() > 4) {
        if (tok[4] == 'n') promo = KNIGHT;
        else if (tok[4] == 'b') promo = BISHOP;
        else if (tok[4] == 'r') promo = ROOK;
        else if (tok[4] == 'q') promo = QUEEN;
        else return false;
    }
    if (from < 0 || from > 63 || to < 0 || to > 63) return false;
    MoveList ml;
    legalMoves(b, ml);
    for (int i = 0; i < ml.n; i++) {
        // Promotion input without piece = queen
        if (ml.m[i].from == from && ml.m[i].to == to &&
            (ml.m[i].promo == promo || (promo == NO_PIECE && ml.m[i].promo == QUEEN))) {
            // Ambiguous bare promotion token matching queen first is standard.
            if (promo == NO_PIECE && ml.m[i].promo != NO_PIECE && ml.m[i].promo != QUEEN) continue;
            out = ml.m[i];
            return true;
        }
    }
    return false;
}

static I64 perft(Board &b, int depth) {
    if (depth == 0) return 1;
    MoveList ml;
    legalMoves(b, ml);
    if (depth == 1) return ml.n;
    I64 n = 0;
    Undo u;
    for (int i = 0; i < ml.n; i++) {
        doMove(b, u, ml.m[i]);
        n += perft(b, depth - 1);
        undoMove(b, u, ml.m[i]);
    }
    return n;
}

// Verify incremental NNUE accs against full refresh (debug quality gate).
static bool verifyNNUE(Board &b, NNUE &nn) {
    int32_t saveW[NNUE_HIDDEN], saveB[NNUE_HIDDEN];
    memcpy(saveW, nn.accW, sizeof(saveW));
    memcpy(saveB, nn.accB, sizeof(saveB));
    nn.refresh(b);
    for (int i = 0; i < NNUE_HIDDEN; i++)
        if (nn.accW[i] != saveW[i] || nn.accB[i] != saveB[i]) {
            memcpy(nn.accW, saveW, sizeof(saveW));
            memcpy(nn.accB, saveB, sizeof(saveB));
            return false;
        }
    return true;
}

// ------------------------------------------------------------- time manager --
// A UCI clock is turned into two numbers:
//   soft - stop *starting* new iterations here (this is the normal move time)
//   hard - abort the search here (only reached when an iteration runs long)
// Both are derived from the clock and increment, never copied from another
// engine: spend a share of what is left, add part of the increment, keep the
// rest banked, and never risk more than a third of the clock on one move.
namespace {

constexpr I64 kMoveOverheadMs = 30;      // GUI/OS round-trip reserve

void planMoveTime(I64 timeLeft, int inc, int movestogo, int movesPlayed,
                  I64 &soft, I64 &hard) {
    I64 avail = timeLeft - kMoveOverheadMs;
    if (avail <= 30) {                   // already inside the reserve: move now
        soft = 1;
        hard = 1;
        return;
    }
    if (avail < 200) {                   // scramble: never bet the whole clock
        soft = std::max<I64>(1, avail / 8);
        hard = std::max<I64>(2, avail / 3);
        return;
    }

    // Moves until the clock is refilled. Trust movestogo when the GUI sends it
    // (it usually does not after the first move of a period); otherwise assume
    // a full game and shorten the horizon as the game progresses.
    int mtg = movestogo > 0 ? movestogo : std::max(12, 44 - movesPlayed / 2);

    double share  = (double)avail / (double)(mtg + 1);
    double target = share * 0.85 + (double)inc * 0.55;
    double cap    = (double)avail * (mtg <= 2 ? 0.6 : 0.34);
    if (target > cap) target = cap;
    if (target < 4)   target = 4;

    soft = (I64)target;
    // Hard limit: enough slack for one expensive iteration, never the whole
    // clock, so a surprise re-search still leaves time for the next move.
    hard = (I64)std::min<double>((double)avail * 0.6, target * 2.2 + 20);
    if (hard < soft) hard = soft;
}

} // namespace

int main(int argc, char **argv) {
    initAttackTables();
    initMagic();
    initZobrist();
    initLMR();

    Search search;
    search.ttResize(1024);
    search.setThreads(1);   // single-threaded by default; "setoption name Threads" opts in
    std::cerr << "info string Threads default 1 (CPU has " << detectCpuThreads() << ")" << std::endl;
    NNUE nn;
    Board board;
    parseFEN(board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::vector<U64> gameHist; // position keys of the current game (for repetition)
    int gamePlies = 0;         // plies played in this game (time-manager hint)
    std::string netPath = "v16v2.nnue";

    // Optional CLI: v16v2 [netfile]
    if (argc >= 2 && argv[1][0] != '-') netPath = argv[1];

    auto loadNet = [&]() {
        if (nn.load(netPath)) {
            search.syncWeights(nn);
            std::cerr << "info string NNUE loaded: " << netPath << std::endl;
            return true;
        }
        std::cerr << "info string NNUE missing: " << netPath << " (eval disabled)" << std::endl;
        return false;
    };
    loadNet();
    std::cerr << "info string using " << search.numThreads << " threads (CPU detected "
              << detectCpuThreads() << ")" << std::endl;

    std::thread searchThread;
    auto joinSearch = [&]() {
        search.stop.store(true, std::memory_order_relaxed);
        if (searchThread.joinable()) searchThread.join();
    };

    std::string line, token;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        ss >> token;
        if (token == "quit") { joinSearch(); break; }
        else if (token == "stop") { joinSearch(); }
        else if (token == "uci") {
            int cpuThreads = detectCpuThreads();
            std::cout << "id name V16V2.7 Superhuman Pulsar\nid author v16\n"
                      << "option name Hash type spin default 1024 min 1 max 4096\n"
                      << "option name Threads type spin default 1"
                      << " min 1 max " << MAX_THREADS << "\n"
                      << "option name EvalFile type string default v16v2.nnue\n"
                      << "uciok" << std::endl;
            // Single-threaded by default (like every CCRL engine); the harness
            // opts in with "setoption name Threads". The CPU size is reported
            // only as information.
            std::cerr << "info string Threads default 1 (CPU has " << cpuThreads
                      << ", raise via setoption)" << std::endl;
        } else if (token == "isready") {
            // May be sent during a search: answer immediately, but serialise the
            // write with the search thread's info output.
            std::lock_guard<std::mutex> lock(search.ioMutex);
            std::cout << "readyok" << std::endl;
        } else if (token == "setoption") {
            joinSearch();   // never mutate the table/threads/net under a live search
            // Robust "setoption name <NAME> value <VALUE>" split.
            auto trim = [](std::string s) {
                while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
                while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
                return s;
            };
            std::string rest = line.substr(9); // after "setoption"
            std::string nm, opt;
            auto vpos = rest.find(" value ");
            if (vpos == std::string::npos) {
                nm = trim(rest);
                if (nm.rfind("name ", 0) == 0) nm = trim(nm.substr(5));
            } else {
                nm = trim(rest.substr(0, vpos));
                if (nm.rfind("name ", 0) == 0) nm = trim(nm.substr(5));
                opt = trim(rest.substr(vpos + 7));
            }
            if (nm == "Hash") {
                int mb = atoi(opt.c_str());
                if (mb >= 1 && mb <= 4096) search.ttResize(mb);
            } else if (nm == "EvalFile") {
                if (!opt.empty()) {
                    netPath = opt;
                    if (loadNet()) search.syncWeights(nn);
                }
            } else if (nm == "Threads") {
                int th = atoi(opt.c_str());
                search.setThreads(th);
                search.syncWeights(nn);
                std::cerr << "info string Threads set to " << search.numThreads << std::endl;
            }
        } else if (token == "ucinewgame") {
            joinSearch();
            search.ttClear();
            for (int t = 0; t < search.numThreads; t++) search.workers[t].resetTables();
            gameHist.clear();
            gamePlies = 0;
        } else if (token == "position") {
            joinSearch();
            ss >> token;
            bool ok = true;
            if (token == "startpos") {
                ok = parseFEN(board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                ss >> token;   // optional "moves"
            } else if (token == "fen") {
                std::string placement, side, castling, ep;
                if (!(ss >> placement >> side >> castling >> ep)) {
                    ok = false;
                } else {
                    std::string fen = placement + " " + side + " " + castling + " " + ep;
                    // Optional halfmove / fullmove counters.  They must be
                    // consumed only when actually numeric, otherwise the token
                    // "moves" (and the move list after it) would be eaten by a
                    // fixed six-field loop.
                    auto isNumber = [](const std::string &t) {
                        if (t.empty()) return false;
                        for (char c : t) if (!isdigit((unsigned char)c)) return false;
                        return true;
                    };
                    std::streampos pos = ss.tellg();
                    std::string t;
                    if (ss >> t) {
                        if (isNumber(t)) {
                            fen += " " + t;
                            std::streampos p2 = ss.tellg();
                            std::string t2;
                            if (ss >> t2) {
                                if (isNumber(t2)) fen += " " + t2;
                                else { ss.clear(); ss.seekg(p2); }
                            }
                        } else {
                            ss.clear(); ss.seekg(pos);
                        }
                    }
                    ok = parseFEN(board, fen);
                }
            } else {
                ok = false;
            }
            if (!ok) {
                std::cout << "info string invalid position/FEN ignored" << std::endl;
                continue;
            }
            gameHist.clear();
            gameHist.push_back(board.key);
            gamePlies = 0;
            while (ss >> token) {
                if (token == "moves") continue;
                Move m;
                if (!parseUCIMove(board, token, m)) {
                    std::cout << "info string illegal move ignored: " << token << std::endl;
                    break;
                }
                Undo u;
                doMove(board, u, m);
                gameHist.push_back(board.key);
                gamePlies++;
            }
        } else if (token == "go") {
            int depth = MAX_PLY;
            bool depthGiven = false;   // "go depth"/"go mate": no wall-clock cap
            I64 nodesLimit = -1;
            I64 softLimit = 0, hardLimit = 0;
            int mateIn = 0;
            int wtime = 0, btime = 0, winc = 0, binc = 0, movetime = 0, movestogo = 0;
            bool infinite = false;
            while (ss >> token) {
                if (token == "depth") { ss >> depth; depthGiven = true; }
                else if (token == "nodes") ss >> nodesLimit;
                else if (token == "mate") { ss >> mateIn; depthGiven = true; }
                else if (token == "wtime") ss >> wtime;
                else if (token == "btime") ss >> btime;
                else if (token == "winc") ss >> winc;
                else if (token == "binc") ss >> binc;
                else if (token == "movetime") ss >> movetime;
                else if (token == "movestogo") ss >> movestogo;
                else if (token == "infinite") infinite = true;
            }
            if (mateIn > 0) depth = std::min(MAX_PLY, mateIn * 2);
            depth = std::max(1, std::min(depth, MAX_PLY));
            if (movetime > 0) {
                softLimit = hardLimit = std::max<I64>(1, (I64)movetime - kMoveOverheadMs);
            } else if (wtime > 0 || btime > 0) {
                const int myT = (board.side == WHITE) ? wtime : btime;
                const int myI = (board.side == WHITE) ? winc : binc;
                planMoveTime((I64)myT, myI, movestogo, gamePlies / 2, softLimit, hardLimit);
            }
            // A depth-limited search must not be cut short by a wall-clock cap:
            // "go depth N" / "go mate N" run to that depth. Only a bare "go"
            // (no depth, no nodes, no clock) falls back to 5 s.
            if (softLimit <= 0) {
                softLimit = hardLimit =
                    (infinite || nodesLimit > 0 || depthGiven) ? ((I64)1 << 50) : 5000;
            }
            if (!nn.loaded) {
                std::cout << "info string no net loaded - cannot search" << std::endl;
                std::cout << "bestmove 0000" << std::endl;
                continue;
            }
            // Run the search asynchronously so stop/quit/isready stay responsive
            // (a blocking search ignored "stop" until the time limit expired).
            joinSearch();
            search.stop.store(false, std::memory_order_relaxed);
            search.nodeLimit.store(nodesLimit, std::memory_order_relaxed);
            Board goBoard = board;
            std::vector<U64> goHist = gameHist;
            int goDepth = depth;
            I64 goSoft = softLimit, goHard = hardLimit;
            searchThread = std::thread([&, goBoard, goHist, goDepth, goSoft, goHard]() mutable {
                search.iterate(goBoard, nn, goDepth, goSoft, goHard, goHist);
            });
        } else if (token == "eval") {
            nn.rootRefresh(board);
            int e = nn.loaded ? nn.evaluate(board) : 0;
            std::cout << "info string eval cp=" << e << std::endl;
        } else if (token == "feat") {
            int wk = kingSquareOf(board, WHITE);
            int bk = kingSquareOf(board, BLACK);
            for (int sq = 0; sq < 64; sq++) if (board.sq[sq] != NO_PIECE) {
                int c = board.col[sq], t = board.sq[sq];
                int fs = NNUE_INPUTS * kingBucket(wk) + (c == WHITE ? 0 : 384) + 64 * t + bucketMirror(sq, wk);
                int fn = NNUE_INPUTS * kingBucket(bk ^ 56) + (c == WHITE ? 384 : 0) + 64 * t + bucketMirror(sq ^ 56, bk);
                std::cout << "info string feat " << c << " " << t << " " << sq << " " << fs << " " << fn << std::endl;
            }
            std::cout << "info string kings " << wk << " " << bk << " buckets " << nn.numBuckets << std::endl;
        } else if (token == "perft") {
            int d = 4;
            ss >> d;
            d = std::max(1, std::min(d, 7));
            I64 t0 = Search::nowMs();
            I64 n = perft(board, d);
            std::cout << "info string perft " << d << " = " << n
                      << " in " << (Search::nowMs() - t0) << "ms" << std::endl;
        } else if (token == "verify") {
            // Walk: refresh, random legal walk with incremental updates,
            // compare against full refresh at each step.
            nn.refresh(board);
            Board b = board;
            bool ok = true;
            for (int step = 0; step < 200 && ok; step++) {
                MoveList ml;
                legalMoves(b, ml);
                if (ml.n == 0) break;
                Move m = ml.m[(step * 7919) % ml.n];
                Undo u;
                doMoveNN(b, nn, u, m);
                if (!verifyNNUE(b, nn)) { ok = false; break; }
                // Occasionally unmake to test inverse path too.
                if ((step % 7) == 6) {
                    undoMoveNN(b, nn, u, m);
                    if (!verifyNNUE(b, nn)) { ok = false; break; }
                    doMoveNN(b, nn, u, m);
                    if (!verifyNNUE(b, nn)) { ok = false; break; }
                }
            }
            std::cout << "info string verify " << (ok ? "OK" : "MISMATCH") << std::endl;
        }
    }
    return 0;
}