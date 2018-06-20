#!/usr/bin/env python3
import utils.gatepirate as gatepirate
var = gatepirate.ITKGatePirate(port='/dev/ttyACM0', pmode='serial')

var.listen()
