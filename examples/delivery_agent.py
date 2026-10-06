"""A toy courier agent that runs in-process, to try opendial without any services.

uv run opendial run examples/scenarios --target examples.delivery_agent:agent
"""

from opendial.transports import ToolCall
from opendial.transports.loopback import Reply

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday")


async def agent(text: str) -> Reply:
    said = text.lower()
    if not said:
        return Reply("Hello, thanks for calling Swift Parcels. How can I help you today?")
    if said.startswith("[dtmf"):
        order = said.removeprefix("[dtmf ").rstrip("#]")
        return Reply(
            "Thank you, I found your order. What day works for you?",
            tools=[ToolCall("lookup_order", {"order_id": order})],
        )
    for day in DAYS:
        if day in said:
            return Reply(
                f"Done. Your delivery is now on {day.title()}. Anything else?",
                tools=[ToolCall("reschedule", {"day": day})],
            )
    if any(word in said for word in ("bye", "thank", "that's all")):
        return Reply("You're welcome. Goodbye!", hangup=True)
    if any(word in said for word in ("delivery", "package", "parcel", "order")):
        return Reply("Sure. Please enter your order number on your keypad, then press hash.")
    return Reply("Sorry, I didn't catch that. Could you say it again?")
