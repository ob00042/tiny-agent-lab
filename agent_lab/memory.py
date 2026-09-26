from dataclasses import dataclass, field


@dataclass
class MemoryItem:
    kind: str
    content: str


@dataclass
class Memory:
    items: list[MemoryItem] = field(default_factory=list)

    def add(self, kind: str, content: str) -> None:
        self.items.append(MemoryItem(kind=kind, content=content))

    def retrieve(self) -> list[str]:
        return [item.content for item in self.items]

    def retrieve_by_kind(self, kind: str) -> list[MemoryItem]:
        return [item for item in self.items if item.kind == kind]
