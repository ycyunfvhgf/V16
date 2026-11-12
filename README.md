# V16 V2.7 — single-file UCI engine

Everything needed to run is in this folder. The NNUE weights are **linked into
the executable**, so there is no `.nnue` file to ship, copy or configure.

```
v16v2.7-avx2-bmi2-pext.exe   28,070,266 B   AVX2 + BMI2 (PEXT), x86-64-v3  <- default
v16v2.7-avx512.exe           28,082,554 B   AVX-512, x86-64-v4
v16v2.7.exe                  28,070,266 B   same binary as the AVX2 build
README.md                                  this file
books/README.txt                           drop CCRL's opening book here
src/                                       full C++ source + build script (not needed to run)
```

* UCI id: `V16V2.7 Superhuman Pulsar` (author `v16`).
* Platform: **Windows 10 or newer, x86-64**, in two ISA flavours:
  `avx2 ...` needs **AVX2 + BMI2 (PEXT)** — Intel Haswell 2013 / AMD Excavator 2015 or newer;
  `... avx512` needs **AVX-512F/BW/DQ/VL** — Intel Skylake-SP/X 2017 or newer, AMD Zen 4 or newer.
  No AVX-512 in the AVX2 build, no AVX2-only fallback for pre-2013 CPUs.
* Runtime dependencies: **none to ship**. The only imports are `KERNEL32.dll`,
  `ntdll.dll` and the OS-provided UCRT (`api-ms-win-crt-*`), all present on Windows 10+.
  No Visual C++ redistributable, no installer, no registry writes, no temp files.
* No external data: the 25,168,960-byte net is in `.rdata`; 3-piece endgames
  (KQK / KRK / KPK) are generated in-process on demand and reported as `tbhits`.

---

## 1. Quick start

```bat
v16v2.7-avx2-bmi2-pext.exe
:: or, on AVX-512 hardware:
v16v2.7-avx512.exe
```

UCI only — the engine speaks UCI on stdin/stdout and expects a GUI or tournament
manager to drive it. `info string ...` diagnostics go to **stderr**, so they never
pollute the UCI stream.

### Which binary to run

| file | requires | CPUs |
|---|---|---|
| `v16v2.7-avx2-bmi2-pext.exe` | AVX2 + BMI2 (PEXT), `x86-64-v3` | Intel Haswell (2013)+, AMD Excavator/Zen+ |
| `v16v2.7-avx512.exe` | AVX-512F/BW/DQ/VL, `x86-64-v4` | Intel Skylake-SP/X (2017)+, Ice Lake, Tiger Lake, AMD Zen 4+ |
| `v16v2.7.exe` | identical to the AVX2 build | drop-in compatibility name for the web UI |

**Use the AVX2 build unless you know the machine has AVX-512.** Same source,
same net, same search, same strength to the node — the AVX-512 one only lets the
compiler use 512-bit registers for the scalar parts of the code, so it is not
stronger and may be slower on some AVX-512 CPUs that down-clock.

On a CPU missing the required instructions Windows kills the process instantly
with `STATUS_ILLEGAL_INSTRUCTION` (`0xC000001D`); it never plays a move.

### How the binaries were verified

* All three builds give identical score, PV, bestmove **and node counts** on
  8 positions at depth 12.
* The AVX-512 build cannot run on the build machine (i5-8500, no AVX-512 — it
  aborts with `0xC000001D`, as expected), so it was validated under **Intel SDE**
  emulating a Cannon Lake CPU. Compared against the AVX2 build under the same
  emulator: 8/8 identical including node counts, so the code is functionally the
  same, not merely "looks fine". Speed on real AVX-512 hardware is untested.

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

From a normal MinGW-w64 g++ 13 or newer on PATH (tested with 15.2.0):

```bat
cd src
build_one_exe.cmd                       :: -> v16v2.7_all.exe, -march=x86-64-v3
build_one_exe.cmd mynet.nnue out.exe    :: different net / output name
build_one_exe.cmd v16v2.nnue out.exe native   :: tune for the build machine

:: the two released ISA flavours
build_one_exe.cmd v16v2.nnue ..\v16v2.7-avx2-bmi2-pext.exe x86-64-v3
build_one_exe.cmd v16v2.nnue ..\v16v2.7-avx512.exe         x86-64-v4
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

Rough NPS on an i5-8500 (6 cores): **~0.9M single-thread, ~4.7M with 6 threads**.
The node counts above are repeatable to the node at `Threads=1`; wall-clock time
and NPS vary with machine load, so only the node counts are used as a reference.

Search is single-thread deterministic only with `Threads=1` (normal for SMP
engines); use `Threads=1` if you need reproducible games.

## 6. Changelog (useless)

* **2026-11-12** — added the two named ISA builds
  (`v16v2.7-avx2-bmi2-pext.exe`, `v16v2.7-avx512.exe`); verified functionally
  identical to each other and to `v16v2.7.exe`.
* **2026-11-12** — rolled back to the 2026-11-07 search; the 2026-11-09
  overhaul (continuation history, SEE pruning, LMR on losing captures,
  check-extension 10/cap 2) measured ~16% lower NPS and was weaker in play.
  Kept out of this folder.
* **2026-11-11** — net linked into the executable; single-file distribution.
