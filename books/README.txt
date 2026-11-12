OPENING BOOKS GO HERE
=====================

V16 V2.7 has NO internal opening book and never plays a book move.
Opening books belong to the tournament manager / GUI, so CCRL's own book is
used exactly as-is.

Put the book file in this folder (any filename works, any path works — this
folder is only a convenience), then point the GUI at it.

The engine never opens the book, so the name/extension is irrelevant:
CCRL.pgn, book.epd, mybook.bin, whatever.abk — all fine.

WARNING: do not pass the book as a command-line argument to v16v2.7.exe.
The only CLI argument is an optional *net* file. Anything else is parsed as a
net, fails, and the engine runs with its NNUE eval disabled:
    info string NNUE missing: <file> (eval disabled)

cutechess-cli (PGN or EPD book):

    cutechess-cli ^
      -engine cmd=..\v16v2.7.exe name="V16 V2.7" proto=uci ^
      -engine cmd=..\other\engine.exe name="Other" proto=uci ^
      -each tc=40/120 ^
      -openings file=CCRL.pgn format=pgn order=sequential plies=16 ^
      -games 2 -rounds 500 -repeat -pgnout match.pgn

  * format=pgn  - a PGN book (games; the opening lines are replayed)
  * format=epd  - a list of positions
  * plies=16    - both sides get 8 book moves; keep it EVEN so the same opening
                  is played with colours reversed (-games 2 -repeat)
  * order=sequential - play book lines in file order (CCRL convention)
  * order=random     - sample the book randomly

Polyglot (.bin), Arena (.abk), ChessGUI (.cbk):
  cutechess-cli cannot read these. Use a GUI that can (Arena, ChessGUI,
  BanksiaGUI, Fritz): the GUI plays the book move and only sends the engine the
  resulting position, so nothing changes on the engine side.

Nothing needs to be enabled in the engine.
OwnBook / Book / BestBookMove / BookFile options are ignored if sent.

Sanity check: with no GUI book and an empty folder, the engine still plays
1.e4 (or whatever it likes) — that is expected, there is no book to consult.
