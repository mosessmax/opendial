"""The simulated caller: decides what to say, says it over a bad line, hears the reply."""

from __future__ import annotations

import re
from collections.abc import Sequence

from opendial import packs
from opendial.audio.frames import AudioFrame
from opendial.audio.impairments import build_chain
from opendial.providers import LLM, STT, TTS, Message
from opendial.scenario import Scenario

END = "[END]"

_PROMPT = """You are role-playing a phone caller talking to a voice agent.

Who you are: {name}. {description}
Your goal: {goal}
{traits}{context}{accent}
Rules:
- Reply with only what you say out loud, one short turn at a time.
- Only reveal facts from what you know when the agent needs them.
- When your goal is met, or clearly cannot be met, say goodbye and then reply {end} on its own.
"""


def system_prompt(scenario: Scenario) -> str:
    p = scenario.persona
    traits = f"How you behave: {', '.join(p.traits)}.\n" if p.traits else ""
    context = "".join(f"- {k}: {v}\n" for k, v in scenario.context.items())
    context = f"What you know:\n{context}" if context else ""
    accent = ""
    if p.accent:
        pack = packs.accent(p.accent)
        examples = "".join(f"- {e}\n" for e in pack.examples)
        accent = f"How you speak:\n{pack.style}\nFor example:\n{examples}"
    return _PROMPT.format(
        name=p.name,
        description=p.description,
        goal=scenario.goal,
        traits=traits,
        context=context,
        accent=accent,
        end=END,
    )


class Caller:
    def __init__(self, scenario: Scenario, llm: LLM, tts: TTS, stt: STT, seed: int = 0):
        self.scenario = scenario
        self.llm, self.tts, self.stt = llm, tts, stt
        self.line = build_chain(packs.resolve_audio(scenario.audio), seed)
        self._script = list(scenario.script) if scenario.script is not None else None
        self._done = False
        self.history: list[Message] = [{"role": "system", "content": system_prompt(scenario)}]

    async def reply(self, agent_said: str) -> str | None:
        """What the caller says next, or None to hang up."""
        if self._done:
            return None
        if agent_said:
            self.history.append({"role": "user", "content": agent_said})
        if self._script is not None:
            text = self._script.pop(0) if self._script else None
            self._done = not self._script
        else:
            raw = await self.llm.complete(self.history)
            self._done = END in raw  # say the goodbye, hang up on the next turn
            text = raw.replace(END, "").strip() or None
        if text is None:
            self._done = True
            return None
        self.history.append({"role": "assistant", "content": text})
        return text

    @property
    def finished(self) -> bool:
        """True once the caller has said everything it meant to say."""
        return self._done

    async def speak(self, text: str) -> list[AudioFrame]:
        """Synthesize `text` and send it through the impaired line."""
        locale = None
        if self.scenario.persona.accent:
            locale = packs.accent(self.scenario.persona.accent).tts_locale
        return self.line.apply(await self.tts.synthesize(text, locale))

    async def hear(self, frames: Sequence[AudioFrame]) -> str:
        return await self.stt.transcribe(frames)

    def dtmf_for(self, agent_said: str) -> str | None:
        for rule in self.scenario.dtmf:
            if re.search(rule.when, agent_said, re.IGNORECASE):
                return rule.digits
        return None

    def interruption_for(self, agent_turn: int) -> tuple[int, str] | None:
        for i in self.scenario.interruptions:
            if i.turn == agent_turn:
                return i.after_ms, i.say
        return None
