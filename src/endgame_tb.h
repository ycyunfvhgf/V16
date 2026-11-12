// =============================================================================
// endgame_tb.h - exact 3-piece DTM tablebases (KQK, KRK, KPK)
//
// Why this exists:
//   KPK/KQK mates are usually far beyond a 5-second search horizon.  The main
//   search used to score these positions at ~+13 pawns and never announce mate.
//   With an exact distance-to-mate table we can report the true "mate N" and
//   play the optimal move immediately.
//
// How it works:
//   For each material signature (queen/rook/pawn, owner colour) we enumerate
//   every legal position, build the move graph, then run a retrograde
//   (bucket/Dijkstra) analysis:
//       - a side to move with no legal moves is mated (loss, dist 0)
//       - a state is WON if it has a move to a LOST state
//       - a state is LOST if every move leads to a WON state
//       - everything else is a DRAW
//   The distance stored is in plies, from the side-to-move's point of view.
//   KPK promotions are resolved through the KQK/KRK tables, so the full
//   promotion-then-mate line is included in the distance.
//
// The tables are generated lazily, once per material, the first time such a
// position is searched.  They are tiny (a few hundred KB each).
// =============================================================================
#ifndef V16_ENDGAME_TB_H
#define V16_ENDGAME_TB_H

namespace EndgameTB {

static const int TB_N    = 64 * 64 * 64 * 2;  // wk, bk, piece, stm
static const int TB_MAXB = 1024;              // max distance in plies

enum : uint8_t { ST_UNKNOWN = 0, ST_WIN = 1, ST_LOSS = 2, ST_DRAW = 3 };

struct Pos {
    int wk = 0, bk = 0, psq = 0, stm = WHITE;
};

struct TBTable {
    int piece = NO_PIECE;
    int owner = WHITE;
    bool built = false;
    std::vector<uint8_t> st;
    std::vector<int16_t> dtm;
};

// 0 = KQK, 1 = KRK, 2 = KPK; second index = colour that owns the piece.
static TBTable g_tb[3][2];

static inline int tbEnc(const Pos &p) {
    return ((((p.wk << 6) | p.bk) << 6) | p.psq) << 1 | (p.stm & 1);
}
static inline Pos tbDec(int idx) {
    Pos p;
    p.stm = idx & 1; idx >>= 1;
    p.psq = idx & 63; idx >>= 6;
    p.bk  = idx & 63; idx >>= 6;
    p.wk  = idx & 63;
    return p;
}
static inline U64 tbOcc(const Pos &p) {
    return bit(p.wk) | bit(p.bk) | bit(p.psq);
}
static inline int tbKingSq(const Pos &p, int c) {
    return (c == WHITE) ? p.wk : p.bk;
}

// Is "sq" attacked by colour "byColor"?  The only non-king piece belongs to
// "owner" and has type "pt".
static inline bool tbAttackedBy(int sq, int byColor, const Pos &p, int owner, int pt) {
    int bk = tbKingSq(p, byColor);
    if (sq != bk && (kingAttacks[bk] & bit(sq))) return true;
    if (byColor != owner) return false;
    if (pt == QUEEN) {
        U64 occ = tbOcc(p);
        if (bishopAttacks(sq, occ) & bit(p.psq)) return true;
        if (rookAttacks(sq, occ) & bit(p.psq)) return true;
    } else if (pt == ROOK) {
        if (rookAttacks(sq, tbOcc(p)) & bit(p.psq)) return true;
    } else if (pt == PAWN) {
        if (pawnAttacks[owner][p.psq] & bit(sq)) return true;
    }
    return false;
}

static inline bool tbInCheck(const Pos &p, int owner, int pt) {
    return tbAttackedBy(tbKingSq(p, p.stm), p.stm ^ 1, p, owner, pt);
}

// A legal position: distinct squares, kings not adjacent, pawn not on rank
// 1/8, and the side NOT to move is not in check.
static inline bool tbLegal(const Pos &p, int owner, int pt) {
    if (p.wk == p.bk || p.wk == p.psq || p.bk == p.psq) return false;
    if (kingAttacks[p.wk] & bit(p.bk)) return false;
    if (pt == PAWN) {
        int r = rankOf(p.psq);
        if (r == 0 || r == 7) return false;
    }
    int ntmKing = tbKingSq(p, p.stm ^ 1);
    if (tbAttackedBy(ntmKing, p.stm, p, owner, pt)) return false;
    return true;
}

struct GenMove {
    bool promo = false;
    bool capture = false;      // king captured the piece -> KVK draw
    int  pieceType = NO_PIECE;
    Pos  child;
};

// Legal moves for the side to move.  Promotions are emitted as QUEEN and ROOK
// only: bishop/knight promotions can never mate and are draws.
static int tbGenMoves(const Pos &p, int owner, int pt, GenMove *out) {
    int n = 0;
    int stm = p.stm, them = stm ^ 1;

    // ---- king moves ----
    int ksq = tbKingSq(p, stm);
    U64 kt = kingAttacks[ksq];
    while (kt) {
        int t = __builtin_ctzll(kt); kt &= kt - 1;
        if (t == p.wk || t == p.bk) continue;
        bool cap = false;
        if (t == p.psq) {
            if (stm == owner) continue;   // own piece
            cap = true;                   // capturing the only piece -> draw
        }
        Pos c = p; c.stm = them;
        if (stm == WHITE) c.wk = t; else c.bk = t;
        if (cap) {
            int ok = tbKingSq(c, them);
            if (kingAttacks[ok] & bit(t)) continue;   // protected
            GenMove g; g.capture = true; g.child = c;
            out[n++] = g;
            continue;
        }
        if (tbAttackedBy(t, them, c, owner, pt)) continue;
        GenMove g; g.child = c;
        out[n++] = g;
    }

    if (stm != owner) return n;

    // ---- piece moves ----
    if (pt == QUEEN || pt == ROOK) {
        static const int dirs[8][2] = {
            {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}
        };
        int nd = (pt == QUEEN) ? 8 : 4;
        int f0 = fileOf(p.psq), r0 = rankOf(p.psq);
        for (int d = 0; d < nd; d++) {
            int f = f0 + dirs[d][0], r = r0 + dirs[d][1];
            while (f >= 0 && f < 8 && r >= 0 && r < 8) {
                int t = makeSq(r, f);
                if (t == p.wk || t == p.bk) break;    // cannot capture a king
                Pos c = p; c.psq = t; c.stm = them;
                GenMove g; g.child = c;
                out[n++] = g;
                f += dirs[d][0]; r += dirs[d][1];
            }
        }
    } else if (pt == PAWN) {
        int dir = (owner == WHITE) ? 1 : -1;
        int r = rankOf(p.psq), f = fileOf(p.psq);
        int startR = (owner == WHITE) ? 1 : 6;
        int nr = r + dir;
        if (nr >= 0 && nr < 8) {
            int t = makeSq(nr, f);
            if (t != p.wk && t != p.bk) {
                Pos c = p; c.psq = t; c.stm = them;
                if (nr == 0 || nr == 7) {
                    for (int pp : { QUEEN, ROOK }) {
                        GenMove g; g.promo = true; g.pieceType = pp; g.child = c;
                        out[n++] = g;
                    }
                } else {
                    GenMove g; g.child = c;
                    out[n++] = g;
                    if (r == startR) {
                        int mid = makeSq(r + dir, f);
                        int t2  = makeSq(r + 2 * dir, f);
                        if (mid != p.wk && mid != p.bk && t2 != p.wk && t2 != p.bk) {
                            Pos c2 = p; c2.psq = t2; c2.stm = them;
                            GenMove g2; g2.child = c2;
                            out[n++] = g2;
                        }
                    }
                }
            }
        }
    }
    return n;
}

static void tbBuild(int ti, int owner);

static void tbBuild(int ti, int owner) {
    TBTable &T = g_tb[ti][owner];
    if (T.built) return;

    // KPK promotions resolve through KQK/KRK.
    if (ti == 2) { tbBuild(0, owner); tbBuild(1, owner); }

    int pt = (ti == 0) ? QUEEN : (ti == 1) ? ROOK : PAWN;

    std::vector<uint8_t> legal((size_t)TB_N, 0);
    std::vector<int32_t> outdeg((size_t)TB_N, 0);   // edges (normal children)
    std::vector<int32_t> total((size_t)TB_N, 0);    // all legal moves

    // ---- pass A: legality, move counts ----
    for (int idx = 0; idx < TB_N; idx++) {
        Pos p = tbDec(idx);
        if (!tbLegal(p, owner, pt)) continue;
        legal[idx] = 1;
        GenMove gm[64];
        int n = tbGenMoves(p, owner, pt, gm);
        total[idx] = n;
        int edges = 0;
        for (int k = 0; k < n; k++)
            if (!gm[k].capture && !gm[k].promo) edges++;
        outdeg[idx] = edges;
    }

    std::vector<int32_t> fstart((size_t)TB_N + 1, 0);
    for (int i = 0; i < TB_N; i++) fstart[i + 1] = fstart[i] + outdeg[i];
    int64_t E = fstart[TB_N];
    std::vector<int32_t> fedges((size_t)E);

    std::vector<int32_t> remaining((size_t)TB_N, 0);
    std::vector<int32_t> maxWin((size_t)TB_N, 0);
    std::vector<std::vector<int>> lb((size_t)TB_MAXB), pb((size_t)TB_MAXB);

    // ---- pass B: forward edges + promotion bookkeeping ----
    for (int idx = 0; idx < TB_N; idx++) {
        if (!legal[idx]) continue;
        remaining[idx] = total[idx];
        Pos p = tbDec(idx);
        GenMove gm[64];
        int n = tbGenMoves(p, owner, pt, gm);
        int w = fstart[idx];
        for (int k = 0; k < n; k++) {
            GenMove &g = gm[k];
            if (g.capture) continue;              // draw child: never resolves
            if (g.promo) {
                int cti = (g.pieceType == QUEEN) ? 0 : 1;
                TBTable &CT = g_tb[cti][owner];
                int cidx = tbEnc(g.child);
                uint8_t cs = CT.st[cidx];
                int cd = CT.dtm[cidx];
                if (cs == ST_LOSS) {
                    if (cd >= 0 && cd < TB_MAXB) pb[cd].push_back(idx);
                } else if (cs == ST_WIN) {
                    remaining[idx]--;
                    if (cd > maxWin[idx]) maxWin[idx] = cd;
                }
                continue;
            }
            fedges[w++] = tbEnc(g.child);
        }
    }

    // ---- build reverse edges (CSR) ----
    std::vector<int32_t> rstart((size_t)TB_N + 1, 0);
    for (int64_t e = 0; e < E; e++) rstart[fedges[e] + 1]++;
    for (int i = 0; i < TB_N; i++) rstart[i + 1] += rstart[i];
    std::vector<int32_t> redges((size_t)E);
    {
        std::vector<int32_t> cur(rstart.begin(), rstart.end());
        for (int i = 0; i < TB_N; i++)
            for (int32_t e = fstart[i]; e < fstart[i + 1]; e++)
                redges[cur[fedges[e]]++] = i;
    }
    std::vector<int32_t>().swap(fedges);
    std::vector<int32_t>().swap(fstart);

    std::vector<uint8_t> st((size_t)TB_N, ST_UNKNOWN);
    std::vector<int16_t> dtm((size_t)TB_N, 0);

    // ---- terminal states ----
    for (int idx = 0; idx < TB_N; idx++) {
        if (!legal[idx] || total[idx] != 0) continue;
        Pos p = tbDec(idx);
        if (tbInCheck(p, owner, pt)) {
            st[idx] = ST_LOSS; dtm[idx] = 0;
            lb[0].push_back(idx);
        } else {
            st[idx] = ST_DRAW;
        }
    }

    auto assignWin = [&](int pidx, int dist) {
        if (st[pidx] != ST_UNKNOWN) return;
        if (dist <= 0 || dist >= TB_MAXB) return;
        st[pidx] = ST_WIN;
        dtm[pidx] = (int16_t)dist;
        for (int32_t e = rstart[pidx]; e < rstart[pidx + 1]; e++) {
            int q = redges[e];
            if (st[q] != ST_UNKNOWN) continue;
            remaining[q]--;
            if (dist > maxWin[q]) maxWin[q] = dist;
            if (remaining[q] == 0) {
                int qd = maxWin[q] + 1;
                if (qd <= 0 || qd >= TB_MAXB) continue;
                st[q] = ST_LOSS;
                dtm[q] = (int16_t)qd;
                lb[qd].push_back(q);
            }
        }
    };

    // States whose every child is an already-known WIN (promotion wins).
    for (int idx = 0; idx < TB_N; idx++) {
        if (!legal[idx] || st[idx] != ST_UNKNOWN || total[idx] == 0) continue;
        if (remaining[idx] == 0) {
            int qd = maxWin[idx] + 1;
            if (qd > 0 && qd < TB_MAXB) {
                st[idx] = ST_LOSS; dtm[idx] = (int16_t)qd;
                lb[qd].push_back(idx);
            }
        }
    }

    // ---- retrograde, increasing distance ----
    for (int d = 0; d < TB_MAXB; d++) {
        for (size_t k = 0; k < lb[d].size(); k++) {
            int s = lb[d][k];
            for (int32_t e = rstart[s]; e < rstart[s + 1]; e++) {
                int pidx = redges[e];
                if (st[pidx] == ST_UNKNOWN) assignWin(pidx, d + 1);
            }
        }
        for (size_t k = 0; k < pb[d].size(); k++)
            assignWin(pb[d][k], d + 1);
    }

    // Anything still unresolved is a draw.
    for (int idx = 0; idx < TB_N; idx++)
        if (legal[idx] && st[idx] == ST_UNKNOWN) st[idx] = ST_DRAW;

    T.st.swap(st);
    T.dtm.swap(dtm);
    T.piece = pt;
    T.owner = owner;
    T.built = true;
}

static inline int tbKindOfPiece(int pt) {
    return (pt == QUEEN) ? 0 : (pt == ROOK) ? 1 : (pt == PAWN) ? 2 : -1;
}

// Find the single non-king piece on a board.
static bool tbFindPiece(const Board &b, int &owner, int &pt, int &psq) {
    int found = 0;
    for (int s = 0; s < 64; s++) {
        if (b.sq[s] == NO_PIECE || b.sq[s] == KING) continue;
        owner = b.col[s]; pt = b.sq[s]; psq = s;
        found++;
    }
    return found == 1;
}

// Result of a real move, looked up in the appropriate table.
// cs/cd are from the point of view of the side to move in the child position.
static bool tbChild(const Board &b, const Move &m, uint8_t &cs, int &cd) {
    Board nb = b;
    Undo u;
    doMove(nb, u, m);
    int total = __builtin_popcountll(nb.all);
    if (total == 2) { cs = ST_DRAW; cd = 0; return true; }   // piece captured
    int owner, pt, psq;
    if (!tbFindPiece(nb, owner, pt, psq)) { cs = ST_DRAW; cd = 0; return true; }
    int ti = tbKindOfPiece(pt);
    if (ti < 0) { cs = ST_DRAW; cd = 0; return true; }        // B/N is a dead draw
    if (!g_tb[ti][owner].built) tbBuild(ti, owner);
    TBTable &T = g_tb[ti][owner];
    Pos c;
    c.wk = nb.kingSq[WHITE]; c.bk = nb.kingSq[BLACK];
    c.psq = psq; c.stm = nb.side;
    int idx = tbEnc(c);
    cs = T.st[idx]; cd = T.dtm[idx];
    return true;
}

// Probe the root.  Returns true when the position is a 3-piece endgame we can
// solve exactly.  Fills the best move, a score in the engine's mate convention,
// the optimal PV, and the mate distance in plies.
static bool tbProbeRoot(const Board &b, Move &bestOut, int &scoreOut,
                        std::vector<Move> &pvOut, int &matePlies) {
    if (__builtin_popcountll(b.all) != 3) return false;
    int owner, pt, psq;
    if (!tbFindPiece(b, owner, pt, psq)) return false;
    int ti = tbKindOfPiece(pt);
    if (ti < 0) return false;
    tbBuild(ti, owner);

    TBTable &T = g_tb[ti][owner];
    Pos p;
    p.wk = b.kingSq[WHITE]; p.bk = b.kingSq[BLACK];
    p.psq = psq; p.stm = b.side;
    if (!tbLegal(p, owner, pt)) return false;

    int rootIdx = tbEnc(p);
    uint8_t rootSt = T.st[rootIdx];
    int rootDtm = T.dtm[rootIdx];

    // Reconstruct the optimal line.
    Board cur = b;
    std::vector<Move> line;
    for (int guard = 0; guard < 256; guard++) {
        int cowner, cpt, cpsq;
        if (!tbFindPiece(cur, cowner, cpt, cpsq)) break;
        int cti = tbKindOfPiece(cpt);
        if (cti < 0) break;
        if (!g_tb[cti][cowner].built) tbBuild(cti, cowner);
        TBTable &CT = g_tb[cti][cowner];
        Pos cp;
        cp.wk = cur.kingSq[WHITE]; cp.bk = cur.kingSq[BLACK];
        cp.psq = cpsq; cp.stm = cur.side;
        int cidx = tbEnc(cp);
        uint8_t cs = CT.st[cidx];
        int cdtm = CT.dtm[cidx];
        if (cs == ST_LOSS && cdtm == 0) break;   // checkmated: line is complete

        MoveList ml;
        legalMoves(cur, ml);
        if (ml.n == 0) break;

        Move chosen; chosen.from = chosen.to = -1;
        int bestKey = -1000000000;
        for (int i = 0; i < ml.n; i++) {
            uint8_t ks; int kd;
            if (!tbChild(cur, ml.m[i], ks, kd)) continue;
            int key;
            if (cs == ST_WIN) {
                if (ks == ST_LOSS)      key =  1000000 - kd;  // fastest mate
                else if (ks == ST_DRAW) key =  0;
                else                    key = -1000000 - kd;
            } else if (cs == ST_LOSS) {
                if (ks == ST_WIN)       key =  kd;            // longest defence
                else if (ks == ST_DRAW) key =  2000000;
                else                    key = -2000000;
            } else { // DRAW: prefer keeping the draw
                if (ks == ST_DRAW)      key =  2000000;
                else                    key = -kd;
            }
            if (key > bestKey) { bestKey = key; chosen = ml.m[i]; }
        }
        if (chosen.from < 0) break;
        line.push_back(chosen);
        Undo u; doMove(cur, u, chosen);
    }

    pvOut = line;
    if (!line.empty()) bestOut = line[0];
    matePlies = rootDtm;
    if (rootSt == ST_WIN)       scoreOut = MATE_SCORE - rootDtm;
    else if (rootSt == ST_LOSS) scoreOut = -MATE_SCORE + rootDtm;
    else                        scoreOut = 0;
    return true;
}

} // namespace EndgameTB

#endif // V16_ENDGAME_TB_H
