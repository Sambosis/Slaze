# Project Knowledge

## Tech Stack
Django, Flask, FastAPI, Express, Docker

## Architecture
Slazy is a unknown project with 335 files (37,917 lines of code). Tech stack: Django, Flask, FastAPI, Express, Docker. Contains 301 classes and 5277 functions across 2 language(s).

## Key Files & Directories
**Entry Points:**
- `wsgi.py`
- `templates\index.html`
- `repo\3d Sim\app.js`
- `repo\3d Sim\index.html`
- `repo\airhockey\main.py`
- `repo\BattleShip\main.py`
- `repo\BlackJackRL\main.py`
- `repo\bogame\main.py`
- `repo\dots\main.py`
- `repo\glbj\main.py`

**Key Files:**
- `tools\base.py` — Core module with multiple classes
- `repo\maze\character.py` — Complex module with extensive API
- `repo\warlearn\main.py` — Complex module with extensive API
- `repo\BlackJackRL\game\strategy.py` — Core module with multiple classes
- `repo\bogame\snake.py` — Complex module with extensive API
- `tools\cst_code_editor.py` — Core module with multiple classes *(complex)*
- `repo\sol\test_rl_agent.py` — Test suite *(complex)*
- `agent.py` — Core module with multiple classes *(complex)*
- `utils\file_logger.py` — Core module with multiple classes *(complex)*
- `tools\write_code.py` — Core module with multiple classes *(complex)*
- `repo\BlackJackRL\game\blackjack.py` — Core module with multiple classes *(complex)*
- `agent_test.py` — Test suite *(complex)*
- `tools\edit.py` — Core module with multiple classes *(complex)*
- `repo\BattleShip\agents.py` — Core module with multiple classes *(complex)*
- `repo\grokgame\main.py` — Core module with multiple classes *(complex)*

## Build & Run Commands
```bash
python manage.py runserver    # Start dev server
python manage.py migrate      # Apply migrations
python manage.py test         # Run tests
```
```bash
npm run dev     # Start dev server
npm run build   # Production build
npm test        # Run tests
```
```bash
go run .        # Run the project
go build        # Build
go test ./...   # Run tests
```
```bash
pip install -r requirements.txt   # Install deps
python app.py                     # Start server
```

## Conventions
_Conventions will be learned as the AI works with this project._

## Dependencies
_Not yet detected._

## Recent Changes
- Added a full README with setup, run commands, CLI examples, config defaults, and artifact notes.
- Updated `main.py` argparse to use `ArgumentDefaultsHelpFormatter` and removed hardcoded `(default: ...)` text from help strings.
- Fixed `visualizer.py` example block to correctly unpack `init_pygame` return values and call `draw_game` with valid arguments (`clock` instead of invalid `fps`).
