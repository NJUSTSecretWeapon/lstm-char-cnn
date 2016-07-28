local ResMLP = {}

function ResMLP.mlp(size, num_layers, bias, f)
    -- size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- bias = bias for transform gate (default = -2)
    -- f = non-linearity (default = ReLU)
    
    local output, transform_gate, carry_gate, nx
    local num_layers = num_layers or 1
    local bias = bias or -2
    local f = f or nn.ReLU()
    local input = nn.Identity()()
    
    local inputs = {[1] = input}
    if num_layers <= 0 then num_layers = 1 print('num_layers for resmlp in lstm must > 0') end
    for i = 1, num_layers do
        if i == 1 then
            output = f(nn.Linear(size, 4 * size)(inputs[i]))
            --nx     = nn.JoinTable(2)({inputs[i], inputs[i], inputs[i], inputs[i]})
            nx     = nn.Reshape(4 * size, true)(nn.Replicate(4, 2)(inputs[i]))
        else
            output = f(nn.Linear(4 * size, 4 * size)(inputs[i]))
            nx     = inputs[i]
        end
        output = nn.CAddTable()({output, nx})
        table.insert(inputs, output)
    end
    return nn.gModule({input}, {output})
end

--[[reference down --
    local inputs = {[1]=input}
    for i = 1, num_layers do        
        output = f(nn.Linear(size, size)(inputs[i]))
        transform_gate = nn.Sigmoid()(nn.AddConstant(bias)(nn.Linear(size, size)(inputs[i])))
        carry_gate = nn.AddConstant(1)(nn.MulConstant(-1)(transform_gate))
        output = nn.CAddTable()({
	       nn.CMulTable()({transform_gate, output}),
	       nn.CMulTable()({carry_gate, inputs[i]})	})
        table.insert(inputs, output)
    end
    return nn.gModule({input},{output})
end]]

return ResMLP
