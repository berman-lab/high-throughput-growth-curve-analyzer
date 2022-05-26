from dataclasses import dataclass, field

@dataclass(kw_only=True, slots=True)
class ExperimentData:
    times : list[float] = field(default_factory=list)
    temps : list[float] = field(default_factory=list)
    plate_name : str
    file_name : str
    wells: dict = field(default_factory=dict) 