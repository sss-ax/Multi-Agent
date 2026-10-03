"""In-process message transport for native multi-agent workflows.

The optimized workflow uses GraphStore as its semantic transport.  The native
workflow intentionally keeps natural-language payloads, so it needs a real
message boundary instead of passing role fields around directly.  This module
provides deterministic envelopes, per-recipient mailboxes, direct delivery,
broadcast, acknowledgement and a replayable transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Iterable, Mapping


@dataclass
class MessageEnvelope:
    """One immutable-in-content message with mutable delivery receipts."""

    message_id: str
    conversation_id: str
    sender_id: str
    recipient_ids: tuple[str, ...]
    content: str
    message_type: str = "text"
    reply_to: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    delivery: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "sender_id": self.sender_id,
            "recipient_ids": list(self.recipient_ids),
            "content": self.content,
            "message_type": self.message_type,
            "reply_to": self.reply_to,
            "correlation_id": self.correlation_id,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "delivery": dict(self.delivery),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MessageEnvelope":
        recipients = tuple(str(value) for value in payload.get("recipient_ids", ()))
        delivery = {
            str(agent_id): str(status)
            for agent_id, status in dict(payload.get("delivery", {})).items()
        }
        return cls(
            message_id=str(payload["message_id"]),
            conversation_id=str(payload["conversation_id"]),
            sender_id=str(payload["sender_id"]),
            recipient_ids=recipients,
            content=str(payload.get("content", "")),
            message_type=str(payload.get("message_type", "text")),
            reply_to=payload.get("reply_to"),
            correlation_id=payload.get("correlation_id"),
            metadata=dict(payload.get("metadata", {})),
            created_at=float(payload.get("created_at", 0.0)),
            delivery=delivery,
        )


class MessageBusError(ValueError):
    """Raised when a message cannot be routed or acknowledged."""


class MessageBus:
    """Deterministic in-process transport with one mailbox per actor."""

    def __init__(self, conversation_id: str) -> None:
        if not conversation_id.strip():
            raise ValueError("conversation_id must be non-empty")
        self.conversation_id = conversation_id
        self._actors: set[str] = set()
        self._messages: list[MessageEnvelope] = []
        self._next_message_number = 1

    @property
    def actors(self) -> tuple[str, ...]:
        return tuple(sorted(self._actors))

    def register(self, actor_id: str) -> None:
        actor_id = str(actor_id).strip()
        if not actor_id:
            raise MessageBusError("actor_id must be non-empty")
        self._actors.add(actor_id)

    def unregister(self, actor_id: str) -> None:
        self._actors.discard(actor_id)

    def _validate_route(self, sender_id: str, recipients: Iterable[str]) -> tuple[str, ...]:
        sender_id = str(sender_id)
        if sender_id not in self._actors:
            raise MessageBusError(f"unknown sender: {sender_id}")
        unique = tuple(dict.fromkeys(str(value) for value in recipients))
        if not unique:
            raise MessageBusError("message must have at least one recipient")
        unknown = [recipient for recipient in unique if recipient not in self._actors]
        if unknown:
            raise MessageBusError(f"unknown recipient(s): {', '.join(unknown)}")
        return unique

    def send(
        self,
        sender_id: str,
        recipient_id: str,
        content: str,
        *,
        message_type: str = "text",
        reply_to: str | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MessageEnvelope:
        """Deliver one direct message to one recipient mailbox."""
        return self._create(
            sender_id,
            (recipient_id,),
            content,
            message_type=message_type,
            reply_to=reply_to,
            correlation_id=correlation_id,
            metadata=metadata,
        )

    def broadcast(
        self,
        sender_id: str,
        content: str,
        *,
        recipients: Iterable[str] | None = None,
        message_type: str = "broadcast",
        reply_to: str | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MessageEnvelope:
        """Deliver one message to a recipient set as one broadcast event."""
        target_ids = self.actors if recipients is None else tuple(recipients)
        target_ids = tuple(actor_id for actor_id in target_ids if actor_id != sender_id)
        return self._create(
            sender_id,
            target_ids,
            content,
            message_type=message_type,
            reply_to=reply_to,
            correlation_id=correlation_id,
            metadata=metadata,
        )

    def _create(
        self,
        sender_id: str,
        recipients: Iterable[str],
        content: str,
        *,
        message_type: str,
        reply_to: str | None,
        correlation_id: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> MessageEnvelope:
        recipient_ids = self._validate_route(sender_id, recipients)
        message = MessageEnvelope(
            message_id=f"{self.conversation_id}:message:{self._next_message_number}",
            conversation_id=self.conversation_id,
            sender_id=str(sender_id),
            recipient_ids=recipient_ids,
            content=str(content),
            message_type=str(message_type),
            reply_to=reply_to,
            correlation_id=correlation_id,
            metadata=dict(metadata or {}),
            created_at=time.time(),
            delivery={recipient: "pending" for recipient in recipient_ids},
        )
        self._next_message_number += 1
        self._messages.append(message)
        return message

    def receive(
        self,
        actor_id: str,
        *,
        message_types: Iterable[str] | None = None,
        conversation_id: str | None = None,
        limit: int | None = None,
        mark_delivered: bool = True,
    ) -> list[MessageEnvelope]:
        """Read pending messages from an actor's mailbox in send order."""
        self._require_actor(actor_id)
        if limit is not None and int(limit) <= 0:
            return []
        allowed_types = set(message_types) if message_types is not None else None
        messages: list[MessageEnvelope] = []
        for message in self._messages:
            if message.delivery.get(actor_id) != "pending":
                continue
            if conversation_id is not None and message.conversation_id != conversation_id:
                continue
            if allowed_types is not None and message.message_type not in allowed_types:
                continue
            messages.append(message)
            if mark_delivered:
                message.delivery[actor_id] = "delivered"
            if limit is not None and len(messages) >= max(0, int(limit)):
                break
        return list(messages)

    def acknowledge(self, actor_id: str, message_ids: Iterable[str]) -> None:
        """Mark messages as processed by one recipient."""
        self._require_actor(actor_id)
        wanted = {str(message_id) for message_id in message_ids}
        found: set[str] = set()
        for message in self._messages:
            if message.message_id not in wanted:
                continue
            if actor_id not in message.delivery:
                raise MessageBusError(f"actor {actor_id} is not a recipient of {message.message_id}")
            if message.delivery[actor_id] not in {"pending", "delivered", "acknowledged"}:
                raise MessageBusError(f"message {message.message_id} is not acknowledgeable for {actor_id}")
            message.delivery[actor_id] = "acknowledged"
            found.add(message.message_id)
        missing = wanted - found
        if missing:
            raise MessageBusError(f"unknown message(s): {', '.join(sorted(missing))}")

    def pending(self, actor_id: str) -> list[MessageEnvelope]:
        return self.receive(actor_id, mark_delivered=False)

    def transcript(self, *, conversation_id: str | None = None) -> list[dict[str, Any]]:
        return [
            message.to_dict()
            for message in self._messages
            if conversation_id is None or message.conversation_id == conversation_id
        ]

    def restore(self, messages: Iterable[Mapping[str, Any]]) -> None:
        """Restore a checkpointed transcript while retaining registered actors."""
        restored = [MessageEnvelope.from_dict(payload) for payload in messages]
        for message in restored:
            if message.conversation_id != self.conversation_id:
                raise MessageBusError(
                    f"message {message.message_id} belongs to {message.conversation_id}, "
                    f"expected {self.conversation_id}"
                )
            self._validate_route(message.sender_id, message.recipient_ids)
        self._messages = restored
        next_number = 1
        prefix = f"{self.conversation_id}:message:"
        for message in restored:
            if message.message_id.startswith(prefix):
                try:
                    next_number = max(next_number, int(message.message_id[len(prefix):]) + 1)
                except ValueError:
                    pass
        self._next_message_number = next_number

    def _require_actor(self, actor_id: str) -> None:
        if actor_id not in self._actors:
            raise MessageBusError(f"unknown actor: {actor_id}")
