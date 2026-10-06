# opendial

Test your voice agents like you test your code.

opendial places simulated phone calls to your voice agent and checks what
happened. Scenarios are YAML: who is calling, what they want, how bad the
line is, when they interrupt, and what counts as success. Run them from the
CLI or as pytest tests, and get JUnit for CI plus an HTML report with the
transcript and a stereo recording of every call.

It is local-first, provider-agnostic and Apache-2.0. It ships with West
African accent and line-quality packs, because most voice agents are never
tested against Nigerian Pidgin on a weak 3G connection from a busy market.

> Status: early. The core loop, assertions, reports, CLI and pytest plugin
> work. The LiveKit transport is a first cut. Pipecat and SIP come next.

## Try it

```sh
git clone https://github.com/mosessmax/opendial && cd opendial
uv sync
uv run opendial run examples/scenarios --target examples.delivery_agent:agent
```

This calls a toy courier agent in-process, with fake speech, so it needs no
API keys. The report lands in `.opendial/report/index.html`.

## A scenario

```yaml
# reschedule-pidgin.scenario.yaml
persona:
  name: Chidi
  accent: ng-pidgin
goal: Move his delivery to Saturday.
context: {order_id: "4417"}
dtmf:
  - {when: "order number", digits: "4417#"}
interruptions:
  - {turn: 0, after_ms: 1200, say: "Abeg wait"}
audio:
  line: ng-market-call
success:
  - {type: task_completed, judge: The delivery was moved to Saturday}
  - {type: tool_called, name: reschedule, args: {day: saturday}}
  - {type: time_to_first_speech, max_ms: 1500, stat: p95}
  - {type: interruption_handled, max_stop_ms: 800}
  - {type: transcription_accuracy, max_wer: 0.2}
  - {type: no_agent_hangup}
```

Without a `script`, the caller is driven by an LLM that plays the persona.
With one, it says the given lines in order, which is fully deterministic.
`opendial schema` prints the JSON Schema for editor completion, and
`opendial validate` checks files.

## Pointing it at your agent

opendial takes `module:attribute` import paths:

| Flag | Expects |
| --- | --- |
| `--target` | An async `agent(text) -> Reply` (runs in-process) or a factory returning a transport, such as `opendial.transports.livekit:from_env`. |
| `--llm` | Factory for the caller's LLM. Needed unless every scenario has a script. |
| `--tts`, `--stt` | Factories for the caller's voice and ears. Default to fakes, which only work with in-process agents. |
| `--judge` | Factory for the LLM that grades `task_completed`. Without it, that check is skipped. |

A provider is any object with the right async method:

```python
class LLM(Protocol):
    async def complete(self, messages: list[Message]) -> str: ...


class TTS(Protocol):
    async def synthesize(self, text: str, locale: str | None = None) -> list[AudioFrame]: ...


class STT(Protocol):
    async def transcribe(self, frames: Sequence[AudioFrame]) -> str: ...
```

### LiveKit

```sh
uv add "opendial[livekit]"
export LIVEKIT_URL=wss://... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
export OPENDIAL_AGENT_NAME=my-agent   # optional: dispatch a named agent per room
opendial run scenarios --target opendial.transports.livekit:from_env --llm ... --tts ... --stt ...
```

To check tool calls and transcription accuracy, the agent publishes events on
the `opendial` data topic with `opendial.transports.livekit.event_payload`.

## pytest

Every `*.scenario.yaml` is collected as a test, and scenario tags become
markers (`pytest -m smoke`).

```toml
# pyproject.toml
[tool.pytest.ini_options]
opendial_target = "my_agent:agent"
opendial_llm = "my_providers:caller_llm"
```

```python
def test_reschedule(opendial_call):
    outcome = opendial_call("scenarios/reschedule-pidgin.scenario.yaml")
    assert outcome.passed
    assert outcome.record.latencies_ms[0] < 1200
```

Add `--opendial-report out/` for the HTML and JUnit report.

## Packs

```
$ opendial packs
line    ng-market-call       Mobile call from a busy open-air market. Loud crowd noise, light loss.
line    ng-mobile-3g         Narrowband mobile call with codec artefacts and bursty packet loss.
accent  ng-pidgin            Caller speaks Nigerian Pidgin, switching to plain English for numbers and names.
line    ng-poor-signal       Edge-of-coverage call. Heavy, long loss bursts that drop whole words.
accent  ng-yoruba-english    Caller speaks Nigerian English and drops into Yoruba words and tags mid-sentence.
```

Accent packs shape how the caller's LLM speaks, pick a TTS locale, and list
spelling variants (`a beg` and `abeg`) that should not count as
transcription errors. Line packs are named audio conditions. Other packages
can ship their own through the `opendial.packs` entry point.

## How it works

See [docs/architecture.md](docs/architecture.md).

## Development

```sh
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org).

## License

Apache-2.0
