from dataclasses import dataclass, field

@dataclass()
class ExperimentData:
    ODs : field(default_factory=dict)
    times : field(default_factory=list)
    temps : field(default_factory=list)
    plate_name : str
    file_name : str

    # Growth curve parameters estimations
    begin_exponent_time : field(default_factory=dict)
    # maximum population growth rate - denoted by 'a' sometimes
    # under the hood: (x, y, slope)
    max_population_gr : field(default_factory=dict)
    # maximal poplution density -  denoted by k sometimes
    max_population_density : field(default_factory=dict)