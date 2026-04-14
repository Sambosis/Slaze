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
- Added optional per-visualized-episode MP4 recording via `recorder.py`, `train.py`, and `main.py` flags (`--record`, `--record_dir`, `--record_fps`, `--record_frame_skip`).
- Added optional Git sync after each completed visualization via `--push_videos`, `--git_remote`, and `--git_branch`; `train.py` now commits/pushes each timestamped episode video without crashing training on Git failures.
- `pyproject.toml` now includes `imageio` and README documents recording plus remote GitHub access workflow.

## Known Product Offerings
- State Industrial Products positions itself as a facility-maintenance partner offering programs built around products + guidance + service, with emphasis on cleaning, sanitation, odor control, drains, warewash, wastewater, air care, equipment, and housekeeping support.
- Useful sales themes from official site: solve recurring facility problems, strong field-rep support, quick shipping (95% within 24 hours), and hospitality-relevant value around odor reduction, drain maintenance, warewash, and housekeeping consistency.
- Public proof points include State Fragrance Cube for rapid odor resolution, Fresh Zone for wastewater odor complaints, Current Issue parts washing fluid, and long-term warewash support for restaurant operations.
- Do not overstate regulatory/spec claims without checking current labels or product sheets first.

## Vertical Messaging Cheat Sheet
## State Product Cheat Sheet by Vertical

### Hotels
- Common problems: lobby/restroom odors, drain issues, housekeeping consistency, warewash reliability, guest-facing cleanliness, back-of-house sanitation.
- Best-fit State categories: odor control, ambient scenting/air care, drain maintenance, warewash/dish service, housekeeping/facility cleaning support, hand care, floor care, laundry.
- Good message angle: protect guest experience, reduce odor complaints, keep kitchens and public areas consistently clean, support staff with easier maintenance routines.
- Safe example language: "We help hotels stay ahead of recurring odor, drain, warewash, and housekeeping issues with products plus local service support."
- Useful public examples: Fragrance Cube, Fresh Zone, D-Stroy, warewash support.

### Healthcare
- Common problems: persistent restroom/common-area odors, drains, housekeeping consistency, hand hygiene support, laundry demands, appearance and safety expectations.
- Best-fit State categories: cleaning & sanitation, odor control, drain maintenance, hand care, laundry, housekeeping support, floor care, water treatment.
- Good message angle: support cleaner, safer-looking facilities; help environmental services teams stay consistent; reduce recurring nuisance problems.
- Safe example language: "State can support healthcare facilities with practical maintenance solutions for odor, drains, sanitation, and housekeeping consistency."
- Important caution: avoid making disinfecting, regulatory, or clinical-efficacy claims unless verified on the exact product label/spec sheet.

### Restaurants / Foodservice
- Common problems: grease and drain buildup, kitchen odors, warewash performance, sanitation consistency, front-of-house cleanliness, restroom odor issues.
- Best-fit State categories: commercial dishwashing/warewash, drain maintenance, odor control, cleaning & sanitation, hand care, floor care.
- Good message angle: keep kitchens moving, improve wash results, reduce drain and odor interruptions, support a cleaner guest-facing environment.
- Safe example language: "We work on recurring kitchen and restroom issues like drains, odors, and warewash consistency so staff can stay focused on service."
- Useful public examples: warewash support, Fresh Zone, Fragrance Cube.

### Property Management
- Common problems: common-area odors, trash room smells, restroom complaints, drain backups, turnover cleaning, appearance issues, wastewater-related nuisance odors.
- Best-fit State categories: odor control, ambient scenting, wastewater/sewage maintenance, drain maintenance, cleaning & sanitation, floor care, housekeeping/facility support.
- Good message angle: improve tenant/resident experience, protect property appearance, reduce repeat complaints, simplify recurring maintenance issues across sites.
- Safe example language: "State helps property teams address recurring odor, drain, wastewater, and cleaning issues with practical solutions and service support."
- Useful public examples: Fresh Zone, Fragrance Cube.

### Industrial / Manufacturing
- Common problems: parts washing, shop odors, floor and restroom upkeep, wastewater or drain issues, general facility cleanliness, uptime-related maintenance needs.
- Best-fit State categories: parts washing/vehicle maintenance, cleaning & sanitation, drain maintenance, wastewater maintenance, odor control, water treatment, specialty maintenance.
- Good message angle: improve uptime, keep facilities cleaner and safer-looking, address recurring maintenance problems, support teams with reliable supply and field service.
- Safe example language: "State supports industrial facilities with maintenance solutions for recurring cleaning, parts washing, odor, drain, and wastewater issues."
- Useful public examples: Current Issue, D-Stroy.

### How to Position State Across All Verticals
- Position State as a facility-maintenance solutions partner, not just a product vendor.
- Lead with recurring problems, not chemistry.
- Emphasize products + guidance + service.
- Keep messaging tied to cleanliness, safety, appearance, uptime, and consistency.
- Use proof-oriented language and local support/service where relevant.
- Do not overstate technical, regulatory, or efficacy claims without verifying the exact product documentation first.

### Quick Discovery Questions by Vertical
- Hotels: "Where do odor or drain complaints show up most often?" "Any warewash or housekeeping consistency issues?"
- Healthcare: "What recurring odor or housekeeping issues take the most staff time?" "Any drain or restroom complaint areas?"
- Restaurants: "Any repeat problems with drains, odors, or dish results?" "Where does cleaning consistency break down during busy periods?"
- Property management: "What complaints come up repeatedly across properties—odor, drains, trash rooms, restrooms, turnover cleaning?"
- Industrial: "What recurring maintenance issues slow the team down—parts washing, drains, odors, wastewater, or general facility cleaning?"
