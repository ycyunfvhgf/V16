# V16 V2.7 — single-file UCI engine

Everything needed to run is in this folder. The NNUE weights are **linked into
the executable**, so there is no `.nnue` file to ship, copy or configure.

```
current/
  v16v2.7.exe          the engine (28 MB, self-contained, net embedded)
  books/               drop CCRL's opening book here (see books/README.txt)
  src/                 full C++ source + build script
  README.md            this file
```

* UCI id: `V16V2.7 Superhuman Pulsar`
* Platform: Windows x86-64, **AVX2 required** (Intel Haswell 2013 / AMD Excavator 2015 or newer).
  Build target is `x86-64-v3`; it runs on any modern CCRL box. No AVX-512, no runtime DLLs
  (only the Windows system libraries), no installer, no registry, no temp files.
* No external data: the 25 MB net is in `.rdata`; 3-piece endgames
  (KQK / KRK / KPK) are generated in-process on demand.

---

## 1. Quick start

```
v16v2.7.exe
```

UCI only — the engine speaks UCI on stdin/stdout and expects a GUI or tournament
manager to drive it. `info string ...` diagnostics go to **stderr**, so they never
pollute the UCI stream.

## 2. UCI options

| Option | Type | Default | Notes |
|---|---|---|---|
| `Hash` | spin | 1024 | MB, 1..4096 |
| `Threads` | spin | auto (detected cores) | 1..32 |
| `EvalFile` | string | `v16v2.nnue` | **Not needed.** Only used if you deliberately want to override the built-in net; if the named file does not exist the engine keeps the embedded net. |

Command line: `v16v2.7.exe [net.nnue]` forces a net from disk. Leave it out for
normal use.

## 3. Opening books — CCRL

**The engine has no internal book and never plays a book move.** That is
deliberate: it means CCRL's own book is used unchanged and identically for both
engines, so the book cannot influence the result. Configure the book in your
tournament manager / GUI, exactly as you would for any bookless engine.

Drop the book in `books/` (or anywhere on disk) and point the GUI at it.
The file name, extension and location are entirely yours — the engine never
opens it, so any name works (`CCRL.pgn`, `book.bin`, `whatever.abk`, ...).

> **Do not pass the book on the command line.** The only command-line argument
> the engine accepts is an optional *net* file (`v16v2.7.exe net.nnue`). A book
> passed there is parsed as a net, fails, and the engine silently runs with the
> NNUE eval disabled (`info string NNUE missing: ... (eval disabled)`).
> Nothing on the command line is needed for normal use.

### cutechess-cli

```bat
cutechess-cli ^
  -engine cmd=v16v2.7.exe name="V16 V2.7" proto=uci ^
  -engine cmd=..\other\engine.exe name="Other" proto=uci ^
  -each tc=40/120 ^
  -openings file=books\CCRL.pgn format=pgn order=sequential plies=16 ^
  -games 2 -rounds 500 -repeat -pgnout match.pgn
```

* `format=pgn` (a PGN book) or `format=epd` (a list of FEN/EPD positions).
* `plies=16` = both sides get 8 moves of book. Keep `plies` **even** so colours
  are reversed out of the same opening (`-games 2 -repeat`).
* `order=sequential` walks the book in order (CCRL convention); `order=random`
  samples it.
* cutechess-cli does **not** read Polyglot `.bin` books. If CCRL's book is
  Polyglot, use a GUI that does (Arena, ChessGUI, BanksiaGUI, Fritz) — the engine
  side is unchanged either way, since the GUI plays the book move and only sends
  the engine the resulting position.

### Other managers

* **Arena / ChessGUI / BanksiaGUI / Fritz**: add `v16v2.7.exe` as a UCI engine,
  then set the engine's book to *your* book, or set the tournament book to it.
  Polyglot `.bin`, Arena `.abk`, ChessGUI `.cbk` all work through the GUI.
* Nothing has to be enabled in the engine. `OwnBook`, `Book`, `BestBookMove`
  etc. are simply ignored if your GUI sends them.

## 4. Rebuilding from source

`src/` contains everything:

```
src/v16v2.7.cpp         engine source (plain, loads v16v2.nnue from disk)
src/v16v2.7_embed.cpp   same engine, loads the linked-in blob (this is what's built)
src/nnue_embed.S        .incbin of v16v2.nnue into .rdata
src/v16v2.nnue          the net (25 MB) — only needed to rebuild, not to run
src/endgame_tb.h        KQK/KRK/KPK generator (compile-time include)
src/build_one_exe.cmd   one-command build
```

From a normal MinGW-w64 g++ 13+ on PATH:

```bat
cd src
build_one_exe.cmd                       :: -> v16v2.7_all.exe, -march=x86-64-v3
build_one_exe.cmd mynet.nnue out.exe    :: different net / output name
build_one_exe.cmd v16v2.nnue out.exe native   :: tune for the build machine
```

Equivalent manual command:

```bat
g++ -O3 -march=x86-64-v3 -std=c++17 -static -o v16v2.7.exe ^
    v16v2.7_embed.cpp nnue_embed.S -pthread
```

To change the embedded net, regenerate `nnue_embed.S` with the new file name in
its `.incbin` line (the build script does this for you) and rebuild. To add a
newer net without touching the source, just run `v16v2.7.exe newnet.nnue`.

## 5. Reference numbers

startpos, Hash 1024, one thread, `go movetime`:

| time | depth | seldepth | nodes |
|---|---|---|---|
| 1 s | 14 | 23 | 706,527 |
| 5 s | 18 | 29 | 4,126,243 |

Rough NPS: ~4.7M single-thread, ~18M with 6 threads on an i5-8500.

Search is single-thread deterministic only with `Threads=1` (normal for SMP
engines); use `Threads=1` if you need reproducible games.

## 6. Changelog

* **2026-11-12** — rolled back to the 2026-11-07 search; the 2026-11-09
  overhaul (continuation history, SEE pruning, LMR on losing captures,
  check-extension 10/cap 2) measured ~16% lower NPS and was weaker in play.
  Kept out of this folder.
* **2026-11-11** — net linked into the executable; single-file distribution.
