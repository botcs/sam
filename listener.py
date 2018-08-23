#!/usr/bin/env python3
import utils.gatepirate as gatepirate
from utils import initDB
initDB()
var = gatepirate.ITKGatePirate(port='auto', pmode='serial')

var.listen()
