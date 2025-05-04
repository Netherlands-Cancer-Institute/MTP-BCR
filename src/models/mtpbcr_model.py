import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


class ModelBase(nn.Module):
    def __init__(self, args, arch='resnet18', from_imagenet=True):
        super(ModelBase, self).__init__()

        # create base encoder
        print("=> creating model '{}'".format(arch))
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(pretrained=from_imagenet)

        self.model = []
        for name, module in net.named_children():
            # if name == 'conv1':
            #     module = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # if isinstance(module, nn.AdaptiveAvgPool2d):
            #     continue
            if isinstance(module, nn.Linear):
                continue
            self.model.append(module)

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class ContinuousPosEncoding(nn.Module):
    def __init__(self, dim, drop=0.1, maxtime=240):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        position = torch.arange(0, maxtime, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(maxtime, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, xs, times):
        ys = xs
        times = times.long()
        for b in range(xs.shape[1]):
            # xxxxx = self.pe[times[b]]
            ys[:, b] += self.pe[times[b]]
        return self.dropout(ys)


class final_fc(nn.Module):
    def __init__(self, args, final_output_feature_dim):
        super(final_fc, self).__init__()

        self.num_classes = args.num_classes
        self.years_of_history = args.years_of_history
        self._risk = nn.Sequential(nn.Linear(final_output_feature_dim, self.num_classes), nn.ReLU(), )
        self._history = nn.Sequential(nn.Linear(final_output_feature_dim, self.years_of_history), nn.ReLU(), )
        self._position_before = nn.Sequential(nn.Linear(final_output_feature_dim, 4), nn.ReLU(), )
        self._position_next = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )
        self._type_before = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )
        self._type_next = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )
        self._age = nn.Sequential(nn.Linear(final_output_feature_dim, 1), nn.ReLU(), )
        self._density = nn.Sequential(nn.Linear(final_output_feature_dim, 4), nn.ReLU(), )  # density(4)
        self._birads = nn.Sequential(nn.Linear(final_output_feature_dim, 7), nn.ReLU(), )  # birads(7)
        self._location_next = nn.Sequential(nn.Linear(final_output_feature_dim, 8), nn.ReLU(), )  # birads(7)

    def forward(self, x):
        pred = {}

        pred['pred_risk'] = self._risk(x)
        pred['pred_history'] = self._history(x)
        pred['pred_position_before'] = self._position_before(x)
        pred['pred_position_next'] = self._position_next(x)
        pred['pred_type_before'] = self._type_before(x)
        pred['pred_type_next'] = self._type_next(x)
        pred['pred_age'] = self._age(x)
        pred['pred_density'] = self._density(x)
        pred['pred_birads'] = self._birads(x)
        pred['pred_location_next'] = self._location_next(x)

        return pred


class final_fc_side_specific(nn.Module):
    def __init__(self, args, final_output_feature_dim):
        super(final_fc_side_specific, self).__init__()

        self.num_classes = args.num_classes
        self.years_of_history = args.years_of_history

        self._risk = nn.Sequential(nn.Linear(final_output_feature_dim, self.num_classes), nn.ReLU(), )
        self._risk_r = nn.Sequential(nn.Linear(final_output_feature_dim, self.num_classes), nn.ReLU(), )
        self._risk_l = nn.Sequential(nn.Linear(final_output_feature_dim, self.num_classes), nn.ReLU(), )

        self._history = nn.Sequential(nn.Linear(final_output_feature_dim, self.years_of_history), nn.ReLU(), )
        self._position_before = nn.Sequential(nn.Linear(final_output_feature_dim, 4), nn.ReLU(), )
        self._position_next = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )
        self._type_before = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )

        self._type_next = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )
        self._type_next_r = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )
        self._type_next_l = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )

        self._age = nn.Sequential(nn.Linear(final_output_feature_dim, 1), nn.ReLU(), )
        self._density = nn.Sequential(nn.Linear(final_output_feature_dim, 4), nn.ReLU(), )  # density(4)
        self._birads = nn.Sequential(nn.Linear(final_output_feature_dim, 7), nn.ReLU(), )  # birads(7)

        self._location_next = nn.Sequential(nn.Linear(final_output_feature_dim, 8), nn.ReLU(), )  # birads(7)
        self._location_next_r = nn.Sequential(nn.Linear(final_output_feature_dim, 8), nn.ReLU(), )  # birads(7)
        self._location_next_l = nn.Sequential(nn.Linear(final_output_feature_dim, 8), nn.ReLU(), )  # birads(7)

    def forward(self, x):
        pred = {}

        pred['pred_risk'] = self._risk(x)
        pred['pred_risk_r'] = self._risk_r(x)
        pred['pred_risk_l'] = self._risk_l(x)

        pred['pred_history'] = self._history(x)
        pred['pred_position_before'] = self._position_before(x)
        pred['pred_position_next'] = self._position_next(x)
        pred['pred_type_before'] = self._type_before(x)

        pred['pred_type_next'] = self._type_next(x)
        pred['pred_type_next_r'] = self._type_next_r(x)
        pred['pred_type_next_l'] = self._type_next_l(x)

        pred['pred_age'] = self._age(x)
        pred['pred_density'] = self._density(x)
        pred['pred_birads'] = self._birads(x)

        pred['pred_location_next'] = self._location_next(x)
        pred['pred_location_next_r'] = self._location_next_r(x)
        pred['pred_location_next_l'] = self._location_next_l(x)

        return pred


class single_side_fc(nn.Module):
    def __init__(self, args, final_output_feature_dim):
        super(single_side_fc, self).__init__()

        self.num_classes = args.num_classes
        self.years_of_history = args.years_of_history

        self._risk = nn.Sequential(nn.Linear(final_output_feature_dim, self.num_classes), nn.ReLU(), )
        self._history = nn.Sequential(nn.Linear(final_output_feature_dim, self.years_of_history), nn.ReLU(), )
        # self._position_before = nn.Sequential(nn.Linear(final_output_feature_dim, 4), nn.ReLU(), )
        # self._position_next = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )
        self._location_before = nn.Sequential(nn.Linear(final_output_feature_dim, 8), nn.ReLU(), )
        self._location_next = nn.Sequential(nn.Linear(final_output_feature_dim, 8), nn.ReLU(), )

        self._type_before = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )
        self._type_next = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )

        # pred_PCR, pred_pT_stage, pred_pN_stage, pred_pM_stage, pred_ER, pred_PR, pred_Her2
        #      2,       5,              4,             2,          2,        2,       2
        self._PCR = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )
        self._pT_stage = nn.Sequential(nn.Linear(final_output_feature_dim, 5), nn.ReLU(), )
        self._pN_stage = nn.Sequential(nn.Linear(final_output_feature_dim, 4), nn.ReLU(), )
        self._pM_stage = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )
        self._ER = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )
        self._PR = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )
        self._Her2 = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )
        # self._age = nn.Sequential(nn.Linear(final_output_feature_dim, 1), nn.ReLU(), )
        # self._density = nn.Sequential(nn.Linear(final_output_feature_dim, 4), nn.ReLU(), )  # density(4)
        # self._birads = nn.Sequential(nn.Linear(final_output_feature_dim, 7), nn.ReLU(), )  # birads(7)

    def forward(self, x):
        pred = {}

        pred['pred_risk'] = self._risk(x)
        pred['pred_history'] = self._history(x)
        # pred['pred_position_before'] = self._position_before(x)
        # pred['pred_position_next'] = self._position_next(x)
        pred['pred_type_before'] = self._type_before(x)
        pred['pred_type_next'] = self._type_next(x)
        # pred['pred_age'] = self._age(x)
        # pred['pred_density'] = self._density(x)
        # pred['pred_birads'] = self._birads(x)
        pred['pred_location_before'] = self._location_before(x)
        pred['pred_location_next'] = self._location_next(x)

        pred['pred_PCR'] = self._PCR(x)
        pred['pred_pT_stage'] = self._pT_stage(x)
        pred['pred_pN_stage'] = self._pN_stage(x)
        pred['pred_pM_stage'] = self._pM_stage(x)
        pred['pred_ER'] = self._ER(x)
        pred['pred_PR'] = self._PR(x)
        pred['pred_Her2'] = self._Her2(x)

        return pred


class exam_fc(nn.Module):
    def __init__(self, args, final_output_feature_dim):
        super(exam_fc, self).__init__()

        self.num_classes = args.num_classes
        self.years_of_history = args.years_of_history

        self._risk = nn.Sequential(nn.Linear(final_output_feature_dim, self.num_classes), nn.ReLU(), )
        self._history = nn.Sequential(nn.Linear(final_output_feature_dim, self.years_of_history), nn.ReLU(), )
        self._position_before = nn.Sequential(nn.Linear(final_output_feature_dim, 4), nn.ReLU(), )
        self._position_next = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )
        self._location_before = nn.Sequential(nn.Linear(final_output_feature_dim, 8), nn.ReLU(), )
        self._location_next = nn.Sequential(nn.Linear(final_output_feature_dim, 8), nn.ReLU(), )
        self._type_before = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )
        self._type_next = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )

        # pred_PCR, pred_pT_stage, pred_pN_stage, pred_pM_stage, pred_ER, pred_PR, pred_Her2
        #      2,       5,              4,             2,          2,        2,       2
        self._PCR = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )
        self._pT_stage = nn.Sequential(nn.Linear(final_output_feature_dim, 5), nn.ReLU(), )
        self._pN_stage = nn.Sequential(nn.Linear(final_output_feature_dim, 4), nn.ReLU(), )
        self._pM_stage = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )
        self._ER = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )
        self._PR = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )
        self._Her2 = nn.Sequential(nn.Linear(final_output_feature_dim, 2), nn.ReLU(), )
        self._age = nn.Sequential(nn.Linear(final_output_feature_dim, 1), nn.ReLU(), )
        self._density = nn.Sequential(nn.Linear(final_output_feature_dim, 4), nn.ReLU(), )  # density(4)
        self._birads = nn.Sequential(nn.Linear(final_output_feature_dim, 7), nn.ReLU(), )  # birads(7)
        self._manufactor = nn.Sequential(nn.Linear(final_output_feature_dim, 3), nn.ReLU(), )  # birads(7)

    def forward(self, x):
        pred = {}

        pred['pred_risk'] = self._risk(x)
        pred['pred_history'] = self._history(x)
        pred['pred_position_before'] = self._position_before(x)
        pred['pred_position_next'] = self._position_next(x)
        pred['pred_type_before'] = self._type_before(x)
        pred['pred_type_next'] = self._type_next(x)
        pred['pred_age'] = self._age(x)
        pred['pred_density'] = self._density(x)
        pred['pred_birads'] = self._birads(x)
        pred['pred_location_before'] = self._location_before(x)
        pred['pred_location_next'] = self._location_next(x)

        pred['pred_PCR'] = self._PCR(x)
        pred['pred_pT_stage'] = self._pT_stage(x)
        pred['pred_pN_stage'] = self._pN_stage(x)
        pred['pred_pM_stage'] = self._pM_stage(x)
        pred['pred_ER'] = self._ER(x)
        pred['pred_PR'] = self._PR(x)
        pred['pred_Her2'] = self._Her2(x)
        pred['pred_manufactor'] = self._manufactor(x)

        return pred


class MTP_BCR_Model(nn.Module):
    def __init__(self, args,):
        super().__init__()

        self.num_classes = args.num_classes
        self.years_of_history = args.years_of_history
        self.pooling = args.pooling
        self.projection_dim = args.projection_dim
        self.num_heads = args.num_heads
        self.feedforward_dim = args.feedforward_dim
        self.drop_transformer = args.drop_transformer
        self.drop_cpe = args.drop_cpe
        self.image_shape = args.image_shape

        if 'resnet50' in args.arch:
            feature_dim = 2048
        elif 'resnet18' in args.arch:
            feature_dim = 512
        else:
            raise ValueError(f"Model architecture {args.arch} is not supported.")

        image_model = ModelBase(args)
        # print(image_model)
        self.image_model = image_model
        self.group_norm = nn.GroupNorm(32, feature_dim)
        # self.projection = nn.Conv2d(feature_dim, self.projection_dim, (1, 1))
        # transformer_dim = self.projection_dim * self.image_shape[0] * self.image_shape[1]
        transformer_dim = self.projection_dim * self.image_shape[0] * self.image_shape[1] * 4
        self.pos_encoding = ContinuousPosEncoding(transformer_dim, drop=self.drop_cpe)
        transformer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            dim_feedforward=self.feedforward_dim,
            nhead=self.num_heads,
            dropout=self.drop_transformer,
        )
        self.transformer = transformer

        if args.use_risk_factors:
            self.tumor_factor_fc = nn.Sequential(nn.Linear(29, 256), nn.Linear(256, 128), nn.ELU(), )
            self.risk_factor_fc = nn.Sequential(nn.Linear(9, 256), nn.Linear(256, 128), nn.ELU(), )
            cc_mlo_mlp_input_feature_dim = feature_dim * 2 + 128
            final_input_feature_dim = (feature_dim * 5) + (self.projection_dim * 4) + 128
        else:
            cc_mlo_mlp_input_feature_dim = feature_dim * 2
            final_input_feature_dim = (feature_dim * 5) + (self.projection_dim * 4)

        final_output_feature_dim = feature_dim + self.projection_dim

        self.cc_mlo_mlp = nn.Sequential(
            nn.Linear(cc_mlo_mlp_input_feature_dim, feature_dim),
            nn.ReLU(), nn.Dropout(), nn.Linear(feature_dim, feature_dim), nn.ReLU(),)

        self.final_ = nn.Sequential(
            nn.Linear(final_input_feature_dim, final_output_feature_dim), nn.ReLU(), nn.Dropout(), )

        self.method = args.method
        if self.method == 'side_specific_4views_mtp_tumor':
            self.final_fc = final_fc_side_specific(args, final_output_feature_dim)
        else:
            self.final_fc = final_fc(args, final_output_feature_dim)

        self.single_side_img_fc = single_side_fc(args, feature_dim)

        self.exam_mlp = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.ReLU(), nn.Dropout(),
            nn.Linear(feature_dim, feature_dim), nn.ReLU(),)
        self.exam_img_fc = exam_fc(args, feature_dim)

        self.projection = nn.Sequential(
            nn.Linear(feature_dim * 5, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(feature_dim * 2, self.projection_dim * 4),
            nn.ReLU(), )

    def _apply_transformer(self, image_feats: torch.Tensor, times, lens):
        B, T, V, C, H, W = image_feats.shape
        image_feats = image_feats.flatten(start_dim=2).permute([1, 0, 2])  # [N, B, V * C * H * W]
        image_feats = self.pos_encoding(image_feats, times)
        image_feats = self.transformer(image_feats)
        return image_feats.permute([1, 0, 2]).reshape([B, T, V, C, H, W])

    def _pool(self, image_feats, lens):
        if self.pooling == "last_timestep":
            pooled_feats = []
            for b, l in enumerate(lens.tolist()):
                pooled_feats.append(image_feats[b, int(l) - 1])
        elif self.pooling == "sum":
            pooled_feats = []
            for b, l in enumerate(lens.tolist()):
                pooled_feats.append(image_feats[b, : int(l)].sum(0))
        else:
            raise ValueError(f"Unkown pooling method: {self.pooling}")

        pooled_feats = torch.stack(pooled_feats)
        pooled_feats = F.adaptive_avg_pool2d(pooled_feats, (1, 1))
        return pooled_feats.squeeze(3).squeeze(2)

    def _exam_based_classify(self, image_feats, lens):
        B, T, V, C, H, W = image_feats.shape
        pooled_feats = torch.zeros(1, C, H, W).cuda()
        # pooled_feats = []
        for b, l in enumerate(lens.tolist()):
            pooled_feats = torch.cat([pooled_feats, image_feats[b, : int(l)].reshape([1 * l * V, C, H, W])], 0)
            # pooled_feats.append(image_feats[b, : int(l)].reshape([1 * l * V, C, H, W]))

        # pooled_feats = torch.cat(pooled_feats, 1)
        pooled_feats = pooled_feats[1:, ]
        pooled_feats = F.adaptive_avg_pool2d(pooled_feats, (1, 1)).reshape([-1, V * C])
        pred = self.exam_img_fc(pooled_feats)
        return pred

    def _single_side_based_classify(self, image_feats, lens):
        B, T, V, C, H, W = image_feats.shape
        pooled_feats = torch.zeros(1, C, H, W).cuda()
        # pooled_feats = []
        for b, l in enumerate(lens.tolist()):
            pooled_feats = torch.cat([pooled_feats, image_feats[b, : int(l)].reshape([1 * l * V, C, H, W])], 0)
            # pooled_feats.append(image_feats[b, : int(l)].reshape([1 * l * V, C, H, W]))
        # pooled_feats = torch.cat(pooled_feats, 1)
        pooled_feats = pooled_feats[1:, ]
        pooled_feats = F.adaptive_avg_pool2d(pooled_feats, (1, 1)).reshape([-1, V * C])
        pred = self.single_side_img_fc(pooled_feats)
        return pred

    def forward(self, images, times, lens, risk_factor_input=None, tumor_infos=None):

        B, T, V, C, H, W = images.shape  # batch_size(B) / time_points(T) / 4 view of mammograms(V) / H / W
        images = images.reshape([B * T * V, C, H, W])
        # repeat image to RGB fitting pretrained resnet
        images = torch.cat([images, images, images], 1)
        # Apply Image Model
        image_feats = self.image_model(images)
        image_feats = F.relu(self.group_norm(image_feats))  # (B * T=6 * V=4, H=32, W=16)
        image_feats = image_feats.reshape([B * T, V, -1])
        # print('image_feats.shape', tumor_infos.shape)  #  for debug
        # single side based features
        if tumor_infos is not None:
            tumor_infos = tumor_infos.reshape([B * T, 2, -1])
            # print('tumor_infos.shape', tumor_infos.shape)  #  for debug
            tumor_infos_r = self.tumor_factor_fc(tumor_infos[:, 0, :])
            tumor_infos_l = self.tumor_factor_fc(tumor_infos[:, 1, :])
            # print('tumor_infos_l.shape', tumor_infos_l.shape)  #  for debug

            image_feats_r = image_feats[:, :2, :].reshape([B * T, -1])
            image_feats_l = image_feats[:, 2:, :].reshape([B * T, -1])
            # print('image_feats_l.shape', image_feats_l.shape)
            image_feats_r = torch.cat([tumor_infos_r, image_feats_r], dim=1)
            image_feats_r = self.cc_mlo_mlp(image_feats_r)

            image_feats_l = torch.cat([tumor_infos_l, image_feats_l], dim=1)
            image_feats_l = self.cc_mlo_mlp(image_feats_l)
        else:
            image_feats_r = image_feats[:, :2, :].reshape([B * T, -1])
            image_feats_r = self.cc_mlo_mlp(image_feats_r)

            image_feats_l = image_feats[:, 2:, :].reshape([B * T, -1])
            image_feats_l = self.cc_mlo_mlp(image_feats_l)

        right_side_based_pred = self._single_side_based_classify(
            image_feats_r.reshape([B, T, 1, -1, *self.image_shape]), lens)
        left_side_based_pred = self._single_side_based_classify(
            image_feats_l.reshape([B, T, 1, -1, *self.image_shape]), lens)

        # exam based features
        exam_feats = self.exam_mlp(torch.cat([image_feats_r, image_feats_l], 1))
        exam_based_pred = self._exam_based_classify(exam_feats.reshape([B, T, 1, -1, *self.image_shape]), lens)

        # Apply transformer
        # image_feats = image_feats.reshape([B * T, -1])   # (B * 6, 4 * 32 * 16)
        image_feats_plus = torch.cat([exam_feats, image_feats.reshape([B * T, -1])], 1)
        image_feats_proj = self.projection(image_feats_plus).reshape([B, T, 1, -1, *self.image_shape])
        image_feats_trans = self._apply_transformer(image_feats_proj, times, lens)

        # Concat and apply classifier
        # image_feats = exam_feats.reshape([B, T, 1, -1, *self.image_shape])
        image_feats_plus = image_feats_plus.reshape([B, T, 1, -1, *self.image_shape])
        image_feats_combined = torch.cat([image_feats_plus, image_feats_trans], dim=3)
        image_feats_pooled = self._pool(image_feats_combined, lens)
        final_feat = image_feats_pooled.view(B, -1)

        if risk_factor_input is not None:
            risk_factor = self.risk_factor_fc(risk_factor_input)
            final_feat = torch.cat([final_feat, risk_factor.type_as(final_feat)], 1)

        # final pred
        final_feat = self.final_(final_feat)
        pred = self.final_fc(final_feat)
        risk = pred['pred_risk']

        return {
            'right_side_based_pred': right_side_based_pred,
            'left_side_based_pred': left_side_based_pred,
            'exam_based_pred': exam_based_pred,
            'final_pred': pred,
            'pred_risk': risk,
        }
