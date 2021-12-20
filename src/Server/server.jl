########################################################################
# Server
########################################################################

abstract type AbstractServer end

include("fedProxServer.jl")
include("ScaffoldServer.jl")
include("fedDCDServer.jl")
