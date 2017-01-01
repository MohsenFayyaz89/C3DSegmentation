--[[
  Script for Training C3D in Torch
  Mohsen Fayyaz -- Sensifai -- Vision Group
--]]

--Main Code

unpack = unpack or table.unpack

require 'nn'
require 'nnlr'
--require 'dpnn'
require 'cunn'
require 'paths'
require 'torch'
require 'cutorch'
require 'optim'
require 'sys'
local cv = require 'cv'
--require 'cv.videoio'
--require 'cv.imgproc'

--require 'ConvLSTM'
--require 'hdf5'
--require 'UntiedConvLSTM'


local function main()
  cutorch.setDevice(1) --CHECK HERE!
  paths.dofile('opts-UCFSport.lua')
  paths.dofile('c3dSeg.lua')
  trainListPath = 'trainList.txt'
  model = c3dModel
  --print(c3dModel)
  
  loadModel = false
  if loadModel ~= true then
    W = torch.load('/home/snf/mohsen/sport1m.t7')
    local method = 'xavier'
    model = require('weight-init')(model, method)
    
    
    convs ={1,4,7,9,12,14,17,19}
    --convs = {}
    for i,v in ipairs(convs) do
      
      --w = torch.load('/home/deepface/Fayyaz/Caffe/'..i..'.t7')
      --b = torch.load('/home/deepface/Fayyaz/Caffe/'..i..'_b.t7')
      --print(#b)
      --b = torch.reshape(b,b:size(5))
      --print(#w)
      --print(w[11][3][2][3][2])
      --w = w:permute(2,3,4,5,1)
      print(#W.modules[v].weight)
      print(#W.modules[v].bias)

      print(#model.modules[1].modules[v].weight)
      --w = torch.zeros(#model.modules[1].modules[v].weight)
      --b = torch.zeros(#model.modules[1].modules[v].bias)
      model.modules[1].modules[v].weight = W.modules[v].weight:clone()
      model.modules[1].modules[v].bias = W.modules[v].bias:clone()
      --print(model.modules[1].modules[v].weight[1][1][1][1][1])
      
    end
    
    fc ={}-- {2,5}
    for i,v in ipairs(fc) do
      
      --w = torch.load('/home/deepface/Fayyaz/Caffe/'..(i+8)..'.t7')
      --b = torch.load('/home/deepface/Fayyaz/Caffe/'..(i+8)..'_b.t7')
      --b = torch.reshape(b,b:size(5))
      --w = torch.reshape(w,w:size(4),w:size(5))
      --w = w:permute(2,1)
      --print(#w)
      --print(#b)
      
      print(#W.modules[v+21].weight)
      print(#W.modules[v+21].bias)

      print(#model.modules[2].modules[v].weight)
      print(#model.modules[2].modules[v].bias)
      
      --w = torch.zeros(#model.modules[2].modules[v].weight)
      --b = torch.zeros(#model.modules[2].modules[v].bias)
      
      --print(torch.sum(model.modules[2].modules[v].weight))
      --print((w:permute(2,1))[3][11])
      model.modules[2].modules[v].weight = W.modules[v+21].weight
      model.modules[2].modules[v].bias = W.modules[v+21].bias

      --print(W.modules[v+21].bias[2])
      --print(model.modules[1].modules[v].bias[2])
      print(torch.sum(model.modules[2].modules[v].weight))
      
    end
    
    W = {}
    collectgarbage()
    
    --dualCriterion = nn.ModuleCriterion(nn.ClassNLLCriterion(), nn.Log())

  -- move everything to gpu
    model:cuda()
    config = {}
    
  end

  if loadModel == true then      
   model = torch.load('model/model_Last.bin')
   config = torch.load('model/config_Last.bin')
 end
 
  dualCriterion = nn.ClassNLLCriterion()--cudnn.SpatialCrossEntropyCriterion()
  cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
  cudnn.verbose = false
  --model = cudnn.convert(model, cudnn)
  opt.train = true
  dualCriterion:cuda()


  criterion = dualCriterion
  
 
 
  -----------------------------------------------------------------------------
  -- Create model or load a pre-trained one
  if opt.modelFile then -- resume training 
    model  = torch.load(opt.modelFile)
    if opt.train then
      config = torch.load(opt.configFile)
    end
  end
  
  if opt.train then
    -----------------------------------------------------------------------------
    
  --Loading Train List  
  trainList = {}
  for line in io.lines(trainListPath) do
    trainList[#trainList+1] = line
  end
  print('Samples: '..#trainList)
  inputData = trainList
  trainSamples = #inputData
  
  --Shuffling Data
  idx = torch.randperm(trainSamples)
  
  --Loading Mean Data
  fmeans = torch.load('fmeans.dat')
      
  params, grads = model:getParameters()
  print('Number of parameters ' .. params:nElement())
  print('Number of grads ' .. grads:nElement())

  local eta = config.eta or opt.eta 
  local momentum = config.momentum or opt.momentum
  local iter  = config.iter or 1
  local lastT  = config.lastT or 1  
  local epoch = config.epoch or 0
  local err  = 0
  
  model:training()
  --model:forget()
  baseWeightDecay = 0.0005
  eta = 0.1
  local learningRates, weightDecays = model:getOptimConfig(eta, baseWeightDecay)
  print('LR:',learningRates[20])
  sgdconf = {learningRates = learningRates, momentum = momentum, weightDecays = weightDecays, learningRate = eta}
    
  print('Start Iter: ', lastT)	
  
  numBatch = 1 --CHECK HERE!!!
  Channels = 3
  Length = 16
  Width = 112
  Height = 112
  
  logger = optim.Logger('Loss.log')
  logger:setNames{'Training Loss'}
  --logger:style{'~'}
  totalFile = io.open('log/total.log','w')
  
  --Sample Data Line
  ln = 0
  
  --Allocating Cuda Memory
  print('dataAllocStart')
  input = torch.CudaTensor(numBatch, Channels, Length, Width, Height)
  print('dataAlloc1')
  target = torch.CudaTensor(numBatch,Length, Width, Height) --GT for Segmentation
  print('dataAlloc2')
  
  logFileList = {}
  
  --Training Main Loop
  for t = lastT,opt.maxIter do
    
    --Batches Loop
    for b = 1,numBatch do
      
      --Going throw samples
      ln = ln + 1
      if(ln > trainSamples) then
        
        ln = 1
        --Learning Rate Decay Policy
        epoch = epoch + 1
        config.epoch = epoch
        eta = eta*math.pow(0.5,epoch/50)
        opt.eta = eta
        learningRates, weightDecays = model:getOptimConfig(eta, baseWeightDecay)
        sgdconf.learningRate = eta
        sgdconf.learningRates = learningRates
        
    
      end
      
      --shuffled data
      lineNumber = idx[ln]
      
      --print('Line: ', lineNumber, '/ ', trainSamples)
      
      --Data File Path
      framesPath = inputData[lineNumber]
      framesPath = framesPath .. 'frames.t7'
      --print(sampleLine)
      
      --Data Label
      gtPath = inputData[lineNumber]
      gtPath = gtPath .. 'gt.t7'
      --print('LABEL: ',label)
      
      --Loading Data
      --print(framesPath)
      frames  = torch.load(framesPath)
      gts  = torch.load(gtPath)
      
      --print(frames:size())
      nF = frames:size(1)
      
      --Looping Frames
      for f=1,nF do
        
        --Looping Channels (For C3D Preparation)
        for c=1,Channels do
          
          input[b][c][f] = (frames[f][c] - fmeans[c])--:div(255.0)
          
        end --End of Channels Loop
        
      end --End of Frames Loop
      target[b] = gts
      --print ('eb')
      
      --logFileList[b] = io.open('log/'..label .. '.log','a')
      
    end --End of Batches Loop
    --print('dataLoad')
    --------------------------------------------------------------------
    -- define eval closure
    local feval = function(params)
      
      model:zeroGradParameters()
      
      local output = model:forward(input)

      
      --f = criterion:updateOutput(output,target)
      local loss = criterion:forward(output,target:view(-1))
      local  dloss_doutput = criterion:backward(output, target:view(-1))
      model:backward(input, dloss_doutput)
      
      return loss,grads
    end
    
    
    
    --Applying SGD
    _,fs = optim.sgd(feval, params, sgdconf)
    
    --model:forget() --PAY ATTT HERE!!!! MOHSEN (LSTM)
    
    --Accumulating err over statIntervals
    err = err + fs[1]
    
    --print(fs,'   ',err)
    --print(fs[1],'   ',err)
    
    --------------------------------------------------------------------
    -- compute statistics / report error
    if math.fmod(t , opt.statInterval) == 0 then
      avgLoss = (err/opt.statInterval)
      print('==> iteration = ' .. t .. ', average loss = ' .. avgLoss .. ' lr '..eta )
      --file:write(err.."\n")
      logger:add{avgLoss}
      --logger:plot()
      --totalFile:write(t ..' '..avgLoss..''..label.."\n")
      err = 0
      
    end
        
    if opt.save and math.fmod(t , opt.saveInterval) == 0 then
      model:clearState()      
      torch.save('model/model_' .. t .. '.bin', model)
      config = {eta = eta, epsilon = epsilon, alpha = alpha, lastT = t ,iter = iter, epoch = epoch}
      torch.save('model/config_' .. t .. '.bin', config)
      
      --Saving Last Models for Power Failure Resume
      torch.save('model/model_Last.bin', model)
      torch.save('model/config_Last.bin', config)
    end
          
    --model:forget()
    --file:close()
  
   end -- End of Training Main Loop

    totalFile:close()
    print ('Training done')
    collectgarbage()
    
  end -- End of Train Phase check

  -------------------------------------------------------------------------
  
end
main()
