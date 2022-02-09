from dataclasses import dataclass, field

@dataclass()
class ExperimentData:
    times : field(default_factory=list)
    temps : field(default_factory=list)
    plate_name : str
    file_name : str
    wells: field(default_factory=dict) 