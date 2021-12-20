from layer import *


class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None,
                 dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32,
                 residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12,
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = MARL_GraphConstruct(in_dim)

        self.seq_length = seq_length
        self.action_embed = nn.Embedding(17, 5, padding_idx=-1)
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(MARL_mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(MARL_mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_linear = nn.Linear(1, out_dim)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=17,
                                    kernel_size=(1, 1),
                                    bias=True)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)
        self.no_prev_flag = torch.zeros(64, 1).to(device)
        self.klloss = torch.nn.KLDivLoss(reduction='sum')
        self.klloss1 = torch.nn.KLDivLoss(reduction='batchmean')
        # self.adp = []

    def forward(self, input, input_prev, idx=None):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        # identify if there isn't a previous input
        self.no_prev_flag = torch.eq(torch.sum(input_prev, dim=[1, 2, 3]), 0)

        input = self.action_embed(input.long()).squeeze().permute(0, 3, 1, 2)
        input_prev = self.action_embed(input_prev.long()).squeeze().permute(0, 3, 1, 2)

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(input)
                    adp_prev = self.gc(input_prev)
                else:
                    adp = self.gc(input)
                    adp_prev = self.gc(input_prev)
            else:
                adp = self.predefined_A
                adp_prev = self.predefined_A
        # adp_prev[torch.logical_not(self.no_prev_flag), ...] = F.softmax(torch.ones(self.num_nodes, self.num_nodes), dim=1).repeat(int(torch.sum(torch.logical_not(self.no_prev_flag).float())), 1, 1).cuda()
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))
            input_prev = nn.functional.pad(input_prev, (self.receptive_field - self.seq_length, 0, 0, 0))

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)
            else:
                x = self.residual_convs[i](x, adp)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                idx = idx.long()
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)

        x = F.relu(self.end_conv_1(x))
        x = F.relu(self.end_linear(x))
        x = self.end_conv_2(x)

        # KLloss = self.klloss(adp[self.no_prev_flag, ...].log().view(-1, 11), adp_prev[self.no_prev_flag, ...].view(-1, 11))
        # KLloss = torch.where(torch.isnan(KLloss), torch.full_like(KLloss, 0.), KLloss)
        # KLloss = KLloss.sum(dim=1).mean()
        # exit(0)
        '''
        KLloss = self.klloss1(adp[torch.logical_not(self.no_prev_flag), ...].log().view(-1, 11),
                             adp_prev[torch.logical_not(self.no_prev_flag), ...].view(-1, 11))

        KLloss2 = self.klloss(adp[torch.logical_not(self.no_prev_flag), ...].log().view(-1, 11),
                              adp_prev[torch.logical_not(self.no_prev_flag), ...].view(-1, 11))
        '''
        KLloss = self.klloss1(adp[torch.logical_not(self.no_prev_flag), ...].log(),
                              adp_prev[torch.logical_not(self.no_prev_flag), ...])

        # print(adp_prev[torch.logical_not(self.no_prev_flag), ...])
        # exit(0)
        # self.adp = adp
        return x, adp, adp_prev, KLloss
