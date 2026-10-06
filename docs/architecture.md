# Architecture

One test is one simulated phone call. A scenario describes the call, a
simulated caller speaks through a degraded line to the agent and listens
back, and everything that happens is recorded in a `CallRecord`. Assertions
and reports are plain functions of that record.

```
scenario.yaml -> Scenario                     packs/ (accents, line profiles)
                    |                                      |
                    v                                      v
          +------------------------- runner ------------------------+
          |  Caller                                   Transport     |
          |  LLM or script  -- noise, band limit,  -> loopback      |
          |  TTS (speak)       codec, packet loss     livekit       | <-> agent
          |  STT (listen)  <-- VAD turn-taking     <- (pipecat, sip)|
          |                    barge-in, DTMF                       |
          |                  timestamped -> CallRecord              |
          +----------------------------+----------------------------+
                                       v
                 assertions (pure)  ->  reports: JUnit, JSON + WAV, HTML
                                       ^
                         entry points: CLI, pytest plugin
```

## Modules

| Module | Role |
| --- | --- |
| `scenario` | Pydantic schema and YAML loader. `opendial schema` prints the JSON Schema. |
| `packs` | Accent packs (LLM style guide, TTS locale, spelling normalisation for WER) and line packs (named audio conditions). Third parties add packs through the `opendial.packs` entry point. |
| `providers` | `LLM`, `TTS` and `STT` protocols, plus deterministic fakes. No vendor SDK is imported by the core. |
| `audio` | 16-bit frames, energy VAD and `Segmenter`, DTMF tones, and the seeded impairment chain (noise, then band limit, then codec, then bursty loss, matching the physical path). |
| `caller` | Builds the caller's prompt from persona, goal, context and accent; speaks through the impaired line; hears the agent. |
| `transports` | One interface: `connect`, `exchange` (one 20 ms frame each way), `send_dtmf`, `drain_events`, `close`. Tool calls, the agent's own transcripts and hangups arrive as events. |
| `runner` | Drives the call in 20 ms ticks: turn-taking, barge-in timing, DTMF, timeouts. Produces the `CallRecord`. |
| `assertions` | One evaluator per criterion. Only `task_completed` needs an LLM (the judge). |
| `report` | JUnit XML for CI, JSON plus stereo WAV for replay, HTML for people. |
| `session`, `cli`, `pytest_plugin` | Resolve `module:attribute` targets and providers, then run scenarios. |

## Time

The runner and transport exchange audio in lockstep, one 20 ms frame per
tick. Live transports pace this in real time. The loopback transport runs as
fast as the CPU allows, so a two-minute call is tested in milliseconds with
exactly the same timings every run.

## Determinism

All randomness (noise, packet loss) is seeded. With the fake providers or a
fixed script, the same scenario and seed produce the same call, bit for bit.
