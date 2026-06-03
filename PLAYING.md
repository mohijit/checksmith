# Playing against Checksmith

Cute Chess (a chess GUI) is bundled in [tools/cutechess-1.4.0-win64/](tools/cutechess-1.4.0-win64/).
You play against the engine through it.

## 0. Build the engine (once, and after any code change)

```powershell
cargo build --release
```

This produces `target\release\checksmith.exe`, which is what the GUI talks to.

## 1. Open Cute Chess

Double-click [play.bat](play.bat), or run `tools\cutechess-1.4.0-win64\cutechess.exe`.

## 2. Register Checksmith (first time only)

1. **Tools → Settings → Engines** tab.
2. Click **Add** (the `+` / "New" button).
3. Fill in:
   - **Name:** `Checksmith`
   - **Command:** browse to `c:\Projects_New\Checksmith\target\release\checksmith.exe`
   - **Working Directory:** `c:\Projects_New\Checksmith`
   - **Protocol:** `UCI`
4. Click **OK** / **Apply**. Cute Chess will ping the engine (`uci`) and should show
   the name `Checksmith 0.1.0`.

Cute Chess remembers this, so you only do it once (until you move the folder).

## 3. Start a game (Human vs Checksmith)

1. **Game → New** (Ctrl+N).
2. Set **White** = *Human*, **Black** = Checksmith (or swap to play Black).
3. Pick a **time control** (e.g. 5 minutes each, or "Time per move" = 3 s for fast games).
4. Click **OK** and play by dragging pieces.

## Tips

- **Engine is too weak / too strong?** Lower or raise the time control. With more
  time the engine searches deeper. (It has no "skill level" option yet — that and a
  much stronger search arrive in later milestones.)
- **Watch it think:** the GUI shows the engine's `info` line — depth, score (in
  centipawns; positive = good for the side to move), and the move it's considering.
- **Engine vs engine:** set both sides to Checksmith to watch it play itself, useful
  for spotting weaknesses.
- **After changing the code:** re-run `cargo build --release`. The GUI picks up the
  new binary automatically next game (no need to re-register).

## Command-line matches (optional)

For automated testing you can use `cutechess-cli.exe` with the bundled
[engines.json](engines.json):

```powershell
cd c:\Projects_New\Checksmith
.\tools\cutechess-1.4.0-win64\cutechess-cli.exe `
  -engine conf=Checksmith -engine conf=Checksmith `
  -each proto=uci st=0.5 -games 2 -pgnout game.pgn
```

This plays Checksmith against itself and writes the moves to `game.pgn`.
