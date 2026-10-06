# Indic Document Intelligence

Photograph or scan a document in an Indian language, and get back clean text, an English
translation, and the ability to ask questions about it in plain language with answers
traced back to the source.

One pipeline runs the whole job:

```
extract -> detect -> correct -> translate -> chunk -> embed -> store -> retrieve -> answer
```

The browser interface, the HTTP API and the command line all call the same pipeline, so
they behave identically.

## Features

- **Recognition** across English and nine Indian languages, from photographs or scans.
- **PDF support**, including scanned PDFs.
- **Language detection**, reporting every language found and how much of the text it covers.
- **Error correction** of recognition mistakes.
- **Translation** into English or any other supported language.
- **Question answering** over everything you upload, with the sources cited. A question
  the documents do not cover is refused rather than guessed at.
- **Local storage**. Nothing leaves the machine except the text sent to the model provider.

### Supported languages

| Code  | Language  | Script     |
|-------|-----------|------------|
| `eng` | English   | Latin      |
| `hin` | Hindi     | Devanagari |
| `ben` | Bengali   | Bengali    |
| `pan` | Punjabi   | Gurmukhi   |
| `guj` | Gujarati  | Gujarati   |
| `ori` | Odia      | Odia       |
| `tam` | Tamil     | Tamil      |
| `tel` | Telugu    | Telugu     |
| `kan` | Kannada   | Kannada    |
| `mal` | Malayalam | Malayalam  |

Devanagari is shared by Hindi, Marathi, Nepali and Sanskrit, so text in that script is
reported with that caveat.

## Requirements

- Python 3.11 or newer
- A Consciousnx API key

Everything else, including the recognition engine and its language data, is installed by
the launcher.

## Running

On Windows:

```bash
run.bat
```

On Linux or macOS:

```bash
./run.sh
```

That is the whole setup. The launcher:

1. finds a suitable Python
2. creates the virtual environment
3. installs the dependencies
4. installs Tesseract if it is missing, then downloads the language data
5. installs the question answering dependencies
6. creates `.env` from the example

then starts the server with the interface open in your browser. Run it again any time —
every step that is already done is skipped, so later starts take a second.

Then open `.env` and set `CONSCIOUSNX_API_KEY`. Recognition and language detection work
without it; correction, translation and question answering need it.

| Option          | Effect                                                          |
|-----------------|-----------------------------------------------------------------|
| `--skip-search` | Skip the embedding dependencies, which are a large download     |
| `--skip-ocr`    | Skip the recognition engine and its language data               |
| `--reinstall`   | Install dependencies again even when they are already present   |
| `--help`        | Show the available options                                      |

The interface is at http://localhost:8000 and the API reference at
http://localhost:8000/docs.

### Language data

The launcher downloads the language data it needs. You can also manage it directly:

```bash
python -m app.ocr.tessdata --list
```

```bash
python -m app.ocr.tessdata --variant best
```

Three data sets are available. `fast` is the default and is usually enough; `standard`
and `best` are larger and slower, but recognise more accurately.

### Question answering

Question answering needs an extra set of dependencies, which are a large one-time
download. The launcher installs them by default; interrupt it, or pass `--skip-search`,
and everything else still runs while search reports itself unavailable.

### Setting it up by hand

If you would rather not use the launcher:

```bash
python -m venv .venv
```

Activate it — `.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
elsewhere — then install the project and create your configuration:

```bash
pip install -e ".[dev,search]"
```

```bash
cp .env.example .env
```

```bash
python run.py
```

`run.py` must be run with the environment's Python. It says so plainly if the
dependencies are missing rather than failing with a traceback.

The interface has four tabs:

- **Process** — upload an image, PDF or text file, choose languages and options, see each stage
- **Library** — everything stored, with full text and the option to delete
- **Ask** — questions answered from the stored documents, with the retrieved sources shown
- **Translate** — quick translation of pasted text

## Command line

The same pipeline, without the browser:

```bash
python -m app.cli ocr scan.jpg --language kan+eng
```

```bash
python -m app.cli add report.jpg --translate
```

```bash
python -m app.cli ask "What is the leave policy?" --show-sources
```

```bash
python -m app.cli translate "ಇದು ಒಂದು ಪರೀಕ್ಷೆ"
```

`python -m app.cli --help` lists every command. Installing the project also provides an
`indic-doc` executable that does the same thing.

## HTTP API

Everything lives under `/api`.

| Method   | Path                             | Purpose                                        |
|----------|----------------------------------|------------------------------------------------|
| `GET`    | `/api/health`                    | Readiness of each dependency                   |
| `GET`    | `/api/languages`                 | Supported languages and what Tesseract has     |
| `POST`   | `/api/documents`                 | Process an upload (image, PDF or text) and store it |
| `GET`    | `/api/documents`                 | List stored documents                          |
| `GET`    | `/api/documents/{id}`            | One document, with every text stage            |
| `DELETE` | `/api/documents/{id}`            | Remove a document and its index                |
| `POST`   | `/api/documents/{id}/reindex`    | Rebuild chunks with different settings         |
| `POST`   | `/api/ocr`                       | Recognise inline base64 data, storing nothing  |
| `POST`   | `/api/ocr/file`                  | Recognise an upload, storing nothing           |
| `POST`   | `/api/translate`                 | Translate text                                 |
| `POST`   | `/api/search/ask`                | Answer a question from stored documents        |

```bash
curl -X POST http://localhost:8000/api/documents \
  -F "file=@scan.jpg" -F "language=kan+eng" -F "translate=true"
```

```bash
curl -X POST http://localhost:8000/api/search/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the notice period?"}'
```

Setting `API_KEY` in `.env` requires an `X-API-Key` header on every `/api` request. It is
unset by default, which suits a tool running on the machine using it. Set it before
exposing the server to a network, and narrow `CORS_ORIGINS` at the same time.

## Configuration

Every setting is read from the environment or `.env`. See `.env.example` for the full
list with defaults. The ones worth knowing:

| Variable                | Default             | Purpose                                     |
|-------------------------|---------------------|---------------------------------------------|
| `CONSCIOUSNX_API_KEY`   | *(required)*        | Provider credential                         |
| `TESSERACT_CMD`         | *(system path)*     | Absolute path to the binary                 |
| `DATABASE_PATH`         | `data/app.db`       | SQLite location                             |
| `DEFAULT_OCR_LANGUAGES` | `eng+kan+tam+tel`   | Languages used when a request omits them    |
| `CHUNK_SIZE`            | `250`               | Words per chunk                             |
| `CHUNK_OVERLAP`         | `50`                | Words repeated between chunks               |
| `RETRIEVAL_TOP_K`       | `3`                 | Chunks retrieved per question               |
| `SIMILARITY_THRESHOLD`  | `0.20`              | Score below which a question is refused     |
| `MAX_UPLOAD_BYTES`      | `26214400`          | Upload size limit                           |
| `MAX_PDF_PAGES`         | `50`                | Most pages read from one PDF                |
| `PDF_RENDER_DPI`        | `200`               | Resolution for rendering scanned PDF pages  |
| `LLM_TIMEOUT_SECONDS`   | `1200`              | Wait allowed for one model response         |
| `LLM_MAX_RETRIES`       | `0`                 | Retries, each repeating the full wait       |
| `API_KEY`               | *(unset)*           | Shared secret for `/api`                    |

## Troubleshooting

**Installing fails with "filename or extension is too long" on Windows.** Paths are
capped at 260 characters by default, and some packages unpack deeply nested files. Move
the project somewhere shorter, such as `C:\projects\indic-doc-intelligence`, and run the
launcher again. Enabling long path support in Windows also works.

**Recognition shows as unavailable.** The Tesseract binary was not found. Run the
launcher, which installs it, or install it yourself and set `TESSERACT_CMD` to the
binary. The application also looks in the usual Windows install locations on its own.

**Every language is listed as not installed.** This means the binary is missing rather
than the language data: the list comes from asking Tesseract what it has, so with no
binary the answer is nothing. Fix the point above first.

**One language is listed as not installed.** Its traineddata file is missing. Fetch it:

```bash
python -m app.ocr.tessdata
```

**Consciousnx shows as not configured.** `CONSCIOUSNX_API_KEY` is empty in `.env`.
Recognition still works; correction, translation and answering do not.

**Requests fail with `device_unreachable`.** The provider returns this when the Cx
desktop application is not running, since `/raw/v1` routes through it. Start it and
retry. The application reports the provider's message and returns the uncorrected text
rather than failing the request, so recognition and detection results are still usable.

**Search shows as loading or unavailable.** Either the embedding model is still
downloading on first run, in which case it is cached afterwards, or
`sentence-transformers` is not installed. Install it with `run.bat` / `./run.sh` without
`--skip-search`, or directly:

```bash
pip install -e ".[search]"
```

**A request seems to hang.** It probably has not: the provider can take several minutes.
The interface shows elapsed time next to a stop control, so you can see it is still
working and end it if you would rather not wait. Lower `LLM_TIMEOUT_SECONDS` to fail
sooner.

**Answers say the context does not contain the answer.** Retrieval scored below
`SIMILARITY_THRESHOLD`. Either the corpus genuinely does not cover the question, or the
chunks are too coarse — try reindexing with a smaller `CHUNK_SIZE` or a different strategy.

## Tests and experiments

`pytest` runs the 114 unit tests. The recognition tests need a Tesseract binary and
skip themselves when there is none; the indexing and answering tests stub the model
and the embedding encoder, so nothing in the suite touches the network.

The `experiments/` directory holds the scripts behind TealSquare technical report
TSQ-TR-2026-02: a page renderer that builds ground-truth pages in all ten languages
from FLORES-200, and one script per experiment (recognition under six image
conditions, a preparation ablation, language detection, language-string selection,
model-based correction, translation, retrieval and answering, two real photographs,
and per-stage timing). Results land in `experiments/results/` as JSON. The model stages use the
Consciousnx credentials from `.env`.

```bash
pip install -e ".[dev,search,experiments]"
```

```bash
python experiments/exp1_build_pages.py
```

```bash
python experiments/exp1_ocr.py
```

The test suite found a sign error in the deskew step (`src/app/ocr/preprocess.py`):
the correction rotated pages further in the direction of their tilt. It is fixed; the
report records the accuracy before and after.

## Credits

Built by:

- **Rishi** — recognition, language detection and correction — [rishiam2005@gmail.com](mailto:rishiam2005@gmail.com)
- **Ganesh** — search and question answering — [ganeshfb2729@gmail.com](mailto:ganeshfb2729@gmail.com)
- **Manasa** — translation — [manasasr85@gmail.com](mailto:manasasr85@gmail.com)
