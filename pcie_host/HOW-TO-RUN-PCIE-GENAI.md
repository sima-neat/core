# Running `pcie-genai` (text and image chat over PCIe)

`pcie-genai` runs a language model on the Modalix card and talks to it from
the host over PCIe. You type a question on the host; the answer streams back
token by token. Some models are VLMs (vision language models): they can also
answer about images. The model remembers the conversation, like the LLiMa
devkit CLI.

The host starts the card program (default `pcie-genai-backend`) on the card
over SSH, waits until the model has loaded over PCIe (this can take a few
minutes), then sends your questions. When the CLI exits, it stops the card
program.

## Before you start

On the host:

- The SiMa PCIe host package is installed and `simaai-mla-daemon@0` is
  running.
- The model folder (with `devkit/` and `elf_files/`) is under the host
  `models` serve root. That root is set in `/etc/simaai/simaai-mla-daemon.conf`,
  section `[serve]`, for example `models = /scratch/simaai/models`. Copy the
  model folder there first.
- For images, the same section also needs a `data` root, for example
  `data = /scratch/simaai/data`.
- See the models the host serves (needs no card):

  ```
  pcie-genai --list
  ```

On the card:

- The card program is installed (`/usr/bin/pcie-genai-backend`, a small start
  script, runs the real program) and `simaai-pep-daemon` is running.
- The host can SSH to the card as root (default `root@10.0.<card-id>.2`; set
  up once with `pcie-setup.sh` from the host package).

## Ask a text question

One question, then exit (good for scripts; the answer goes to stdout, status
and stats to stderr):

```
pcie-genai --model Llama-3.2-3B-Instruct-a16w4 --prompt "Why is the sky blue?"
```

Chat (a `>>> ` prompt; arrow keys edit the line, up-arrow recalls earlier
lines; Ctrl-C stops the current answer; `quit` or Ctrl-D exits). The model
remembers the conversation, so follow-up questions work:

```
pcie-genai --model Llama-3.2-3B-Instruct-a16w4
>>> My name is Ana.
>>> What is my name?
Ana.
>>> clear history
Cleared chat history.
```

Earlier lines are saved in `~/.pcie_genai_history` (last 1000 lines).

## Chat commands

Type `help` to see them:

| Command | What it does |
|---|---|
| `add image <path>` | Adds an image (a path on the host) to your next question (VLM models) |
| `clear history` | Forgets the questions, answers and images; keeps the system prompt |
| `print history` | Prints the conversation as JSON (image paths are your host paths) |
| `set system <text>` | Sets the system prompt and clears the history |
| `clear system` | Removes the system prompt and clears the history |
| `enable-thinking` / `disable-thinking` | Thinking mode on/off (if the model has it); clears the history |
| `help` | Lists the commands |
| `quit` / `exit` | Ends the session and stops the card program |

`clear image` is not a command; use `clear history`.

The history is also cleared when you press Ctrl-C during an answer, when the
model gives no answer, or when a question fails. The CLI prints a line when
that happens. The conversation lives on the card, so it ends when the CLI
exits; the next run starts empty.

## Ask about an image (VLM models)

The model must be a VLM, for example `llava-1.5-7b-hf-a16w4`. A text-only
model refuses the image with a clear error (the conversation is kept).

In a chat, `add image <path>` adds the image to your next question. It then
stays in the conversation, so follow-up questions need no new `add image`.
Every `add image` adds one more image. `clear history` forgets them.

```
pcie-genai --model llava-1.5-7b-hf-a16w4
>>> add image /path/to/sjc.jpg
>>> what is on the image
...
>>> what colors are most common in it
...
```

One question (`--image` can be given more than once):

```
pcie-genai --model llava-1.5-7b-hf-a16w4 --prompt "what is on the image" --image /path/to/sjc.jpg
```

How the image moves: the host copies it into `<data serve root>/pcie-genai/`,
the card pulls it over PCIe **once**, and the host copy is deleted after that
question. The card keeps its copy until the history is cleared or the card
program stops.

Without a `data` serve root, `add image` says
`images need a [serve] 'data' root in <config>`.

## Questions from a file or a pipe

When stdin is not a terminal, each line is one question or command:

```
printf 'My name is Ana.\nWhat is my name?\n' | pcie-genai --model Llama-3.2-3B-Instruct-a16w4
pcie-genai --model Llama-3.2-3B-Instruct-a16w4 < questions.txt
```

- There is no `>>> ` prompt and no line editing.
- The conversation is still remembered.
- The run **stops at the first failed line**, with exit code 1. In a terminal
  chat, the session goes on after an error.

## Options

| Option | Default | What it does |
|---|---|---|
| `--model <name>` | (required) | Model folder under the host `models` serve root |
| `--prompt <text>` | (none) | Ask one question, print the answer, exit |
| `--image <path>` | (none) | With `--prompt`: attach an image; can be repeated |
| `--system-prompt <text>` | model default | Starting system prompt |
| `--max-new-tokens N` | model default | Longest answer, in tokens |
| `--card-host HOST` | `10.0.<card-id>.2` | Card address for SSH |
| `--card-id N` | `0` | Which PCIe card |
| `--user USER` | `root` | SSH user on the card; keep `root` (see Limitations) |
| `--queue 0..3` | `3` | Card queue; the tensor pipeline uses `0` |
| `--card-program NAME` | `pcie-genai-backend` | Runs `/usr/bin/NAME` on the card |
| `--ready-timeout-s N` | `900` | How long to wait for the model to load |
| `--conf <path>` | `/etc/simaai/simaai-mla-daemon.conf` | Host daemon config: serve roots for `--list`, the model check and images |
| `--list` | | Print the models the host serves and exit |
| `-h`, `--help` | | Print the usage and exit |

Thinking starts off; turn it on with the `enable-thinking` command.

## Keep the host and the card in step

The host `pcie-genai` and the card program must come from the same build. A
mismatch never answers wrongly in silence:

- **Card program too old** (no chat support): right after the model loads,
  the CLI stops with "The card backend (...) does not support chat: it is
  older than this pcie-genai. Update it, or pick another card program with
  --card-program."
- **Host too old** (sends the old single-image question): the card refuses the
  question and says the host `pcie-genai` must be updated.

Several card programs can be installed side by side, each with its own start
script in `/usr/bin`; pick one with `--card-program <name>`. The real
program's file name must contain that name, because the host finds the
running card program by it. They all use the same queue, so only one runs at
a time.

## Limitations

- One session at a time per card queue.
- Loading the model takes minutes; the conversation ends when the CLI exits.
- If the card sends nothing for 120 s, the question fails. This also applies
  while a large model thinks before its first token.
- A cancelled `--prompt` run still exits with code 0.
- A wrong `--image` path is found only after the model has loaded.
- Run the card program as `root` (the default `--user`). The card's PCIe
  daemon writes every pulled file as root, so as another user the card
  program cannot delete the pulled model files and images. They stay in the
  card's receive folder (`/tmp/pcie-recv`), and after one such run the next
  model load fails with "no space left" (`rc=-28`).
