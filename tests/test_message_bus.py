import pytest

from workflow_runtime.message_bus import MessageBus, MessageBusError


def test_message_bus_supports_direct_mailbox_delivery_and_acknowledgement():
    bus = MessageBus("conversation-1")
    bus.register("alice")
    bus.register("bob")

    message = bus.send(
        "alice",
        "bob",
        "Please review this proposal.",
        message_type="review_request",
        correlation_id="work-1",
    )
    assert message.sender_id == "alice"
    assert message.recipient_ids == ("bob",)
    assert bus.pending("bob")[0].message_id == message.message_id

    received = bus.receive("bob")
    assert [item.content for item in received] == ["Please review this proposal."]
    assert bus.pending("bob") == []
    bus.acknowledge("bob", [message.message_id])
    assert bus.transcript()[0]["delivery"] == {"bob": "acknowledged"}


def test_message_bus_broadcasts_one_event_to_multiple_mailboxes():
    bus = MessageBus("conversation-2")
    for actor in ("planner", "solver", "critic"):
        bus.register(actor)

    message = bus.broadcast("planner", "A new plan is available.")
    assert set(message.recipient_ids) == {"solver", "critic"}
    assert [item.content for item in bus.receive("solver")] == ["A new plan is available."]
    assert [item.content for item in bus.receive("critic")] == ["A new plan is available."]


def test_message_bus_rejects_unknown_routes():
    bus = MessageBus("conversation-3")
    bus.register("alice")
    with pytest.raises(MessageBusError, match="unknown recipient"):
        bus.send("alice", "missing", "hello")
