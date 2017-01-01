--[[
  Script for Training C3D in Torch
  Mohsen Fayyaz -- Sensifai -- Vision Group
--]]

--Training Code Options

opt = {}
-- general options:
--opt.dir     = 


-- Model parameters:
opt.inputSizeW = 227--64   -- width of each input patch or image
opt.inputSizeH = 227--64   -- width of each input patch or image
opt.eta       = 1e-4 -- learning rate
opt.etaDecay  = 1e-5 -- learning rate decay
opt.momentum  = 0.9  -- gradient momentum
opt.maxIter   = 600000 --max number of updates



 

opt.modelFile = nil
opt.configFile = nil
opt.statInterval = 1 -- interval for printing error
opt.v            = false  -- be verbose
opt.display      = false --true -- display stuff
opt.displayInterval = opt.statInterval
opt.save         = true -- save models
opt.saveInterval = 10000

--if not paths.dirp(opt.dir) then
  -- os.execute('mkdir -p ' .. opt.dir)
--end
