from dataclasses import dataclass, field

@dataclass()
class WellData:
    # Indicates if the well had good enough data to pass the fitting procedures successfully
    is_valid: bool
    ODs: field(default_factory=list)
    # Growth curve parameters estimations
    # The end of the lag phase. It's also the begining of the exponent
    # under the hood: (exponent_begin_time, exponent_begin_OD)
    exponent_begin : field(default_factory=tuple)
    
    # maximum population growth rate - denoted by 'a' sometimes
    # under the hood: (time, OD, slope)
    max_population_gr : field(default_factory=tuple)

    # The end of the exponent phase
    # under the hood: (exponent_end_time, exponent_end_OD)
    exponent_end: field(default_factory=tuple)
    
    # maximal poplution density
    max_population_density : float

    # The ODs if rapid growth had kept going
    exponent_ODs: field(default_factory=list)