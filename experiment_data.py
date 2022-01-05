from dataclasses import dataclass, field

@dataclass()
class ExperimentData:
    ODs : field(default_factory=dict)
    times : field(default_factory=list)
    temps : field(default_factory=list)
    plate_name : str
    file_name : str

    # Growth curve parameters estimations
    
    # The end of the lag phase. It's also the begining of the exponent
    # under the hood: (exponent_begin_time, exponent_begin_OD)
    exponent_begin : field(default_factory=dict)
    
    # The end of the exponent phase
    # under the hood: (exponent_end_time, exponent_end_OD)
    exponent_end: field(default_factory=dict)
    
    # maximum population growth rate - denoted by 'a' sometimes
    # under the hood: (x, y, slope)
    max_population_gr : field(default_factory=dict)
    
    # maximal poplution density
    max_population_density : field(default_factory=dict)

    temp1: field(default_factory=dict)
    temp2: field(default_factory=dict)