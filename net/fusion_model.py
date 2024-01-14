import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder
import torchvision.models as models
from . import model_UGT
from timm.models.vision_transformer import vit_base_patch16_224


def D(p, z):
    # [N, E]
    z = z.detach() # stop gradient
    p = (p / p.norm(dim=-1, keepdim=True)).mean(1)
    z = (z / z.norm(dim=-1, keepdim=True)).mean(1)
    # [N E] [N E] -> [N]
    return (p * z).sum(dim=1).mean() # dot product & batch coeff normalization

def simsiam(v_ft, a_ft):
    D1 = D(v_ft, a_ft)
    D2 = D(a_ft, v_ft)
    return 0.5 * (D1 + D2)

  
class VideoFeature(nn.Module):
    def __init__(self, model_args):
        super(VideoFeature, self).__init__()
        self.model_args = model_args
        self.mapper_in = nn.Linear(4096, 768)
        self.encoder = vit_base_patch16_224(pretrained=True, global_pool='')
        self.encoder.patch_embed = nn.Sequential()
        self.encoder.head = nn.Sequential()
        self.mapper_out = nn.Linear(768, model_args.v_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        if seq_len < 196:
            x = torch.cat([x, torch.zeros((batch_size, 196-seq_len, embed_dim)).to(x.device)],dim=1)
        elif seq_len > 196:
            x = x[:,:196,:]
            seq_len = 196
        x = self.mapper_in(x)
        x = self.encoder(x)
        x = self.mapper_out(x)
        return x[:, :seq_len, :]



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 将输入向量拆分为多个头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 注意力加权求和
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 经过线性变换和残差连接
        x = self.fc(attended_values) + x

        return x


class CrossAttention(nn.Module):  
    def __init__(self, feature_size):  
        super(CrossAttention, self).__init__()  
        self.feature_size = feature_size
        self.query = nn.Linear(feature_size, feature_size)  
        self.key = nn.Linear(feature_size, feature_size)  
        self.value = nn.Linear(feature_size, feature_size)  
  
    def forward(self, query, key, value):  
        query = self.query(query)  
        key = self.key(key)  
        value = self.value(value)  
        attention = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))  
        attention = torch.softmax(attention, dim=-1)  
        output = torch.matmul(attention, value)  
        return output



class Matcher(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Matcher, self).__init__()
        self.linear1 = nn.Linear(input_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class FusionModel(nn.Module):
    def __init__(self, model_args):
        super(FusionModel, self).__init__()

        self.num_heads = model_args.num_heads
        self.layers = model_args.layers
        self.attn_mask = model_args.attn_mask
        output_dim = model_args.output_dim
        self.a_dim, self.v_dim = 1024, model_args.v_dim
        self.attn_dropout = model_args.attn_dropout
        self.relu_dropout = model_args.relu_dropout
        self.res_dropout = model_args.res_dropout
        self.out_dropout = model_args.out_dropout
        self.embed_dropout = model_args.embed_dropout
        self.d_v = model_args.v_dim
        self.hidden_1 = 256
        self.hidden_2 = 512
        self.UGT_dim = model_args.v_dim
        self.UGT_hidden_dim = 1024
        combined_dim = 2 * self.d_v

        self.conv_1d_a = nn.Conv1d(self.a_dim, self.d_v, kernel_size=1, padding=0, bias=False)

        self.UGT = model_UGT.UGT(model_args, self.d_v, self.UGT_dim, self.UGT_hidden_dim)

        self.a_mem_1 = self.transformer_arch(self_type='audio_self')
        self.a_mem_2 = self.transformer_arch(self_type='audio_self')
        self.a_mem_3 = self.transformer_arch(self_type='audio_self')
        self.v_mem_1 = self.transformer_arch(self_type='visual_self')
        self.v_mem_2 = self.transformer_arch(self_type='visual_self')
        self.v_mem_3 = self.transformer_arch(self_type='visual_self')
        self.trans_a_mem = self.transformer_arch(self_type='audio_self', scalar=True)
        self.trans_v_mem = self.transformer_arch(self_type='visual_self', scalar=True)
        self.trans_v_with_a = self.transformer_arch(self_type='visual/audio', pos_emb=True)
        self.trans_a_with_v = self.transformer_arch(self_type='audio/visual', pos_emb=True)

        self.proj_aux1 = nn.Linear(self.d_v, self.hidden_2)
        self.proj_aux2 = nn.Linear(self.hidden_2, self.hidden_1)
        self.proj_aux3 = nn.Linear(self.hidden_1, self.d_v)
        self.out_layer_aux = nn.Linear(self.d_v, output_dim)

        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        self.mapping_layers = VideoFeature(model_args)

        self.matcher = Matcher(self.d_v, 256)


    def transformer_arch(self, self_type='audio/visual', scalar=False, pos_emb=False):
        if self_type == 'visual/audio':
            embed_dim, attn_dropout = self.d_v, 0
        elif self_type == 'audio/visual':
            embed_dim, attn_dropout = self.d_v, 0
        elif self_type == 'audio_self':
            embed_dim, attn_dropout = self.UGT_dim, self.attn_dropout
        elif self_type == 'visual_self':
            embed_dim, attn_dropout = self.UGT_dim, self.attn_dropout
        else:
            raise ValueError("Not a valid network")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=self.layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  scalar=scalar,
                                  pos_emb=pos_emb
                                  )

    def compute_similarity_matrix(self, audio_features, visual_features):
        # Normalize features
        audio_features = F.normalize(audio_features, dim=1)
        visual_features = F.normalize(visual_features, dim=1)
        # Compute similarity matrix
        similarity_matrix = torch.mm(audio_features, visual_features.t())
        return similarity_matrix

    def forward(self, x_aud, x_vid):

        """
        audio, and vision should have dimension [batch_size, seq_len, n_features]
        """

        x_vid = x_vid.squeeze(dim=2)
        x_vid = self.mapping_layers(x_vid)

        x_aud = x_aud.transpose(1, 2)
        x_vid = x_vid.transpose(1, 2)

        # 1-D Convolution visual/audio features
        proj_a_v = x_aud if self.a_dim == self.d_v else self.conv_1d_a(x_aud)
        proj_x_a = proj_a_v.permute(2, 0, 1)
        proj_x_v = x_vid.permute(2, 0, 1)
    
        # UGT
        proj_x_v, proj_x_a = self.UGT(proj_x_a, proj_x_v)
        processed_sim = simsiam(proj_x_v, proj_x_a)
        loss1 = 0.5*(1-processed_sim)

        # # Auxiliary audio network
        h_a1 = self.a_mem_1(proj_x_a)
        h_a2 = self.a_mem_2(h_a1)
        h_a3 = self.a_mem_3(h_a2)
        h_rep_a_aux = h_a3[:,-1,:]

        # # Auxiliary visual network
        h_v1 = self.v_mem_1(proj_x_v)
        h_v2 = self.v_mem_2(h_v1)
        h_v3 = self.v_mem_3(h_v2)
        h_rep_v_aux = h_v3[:,-1,:]

        # Calculate hard negatives.
        similarity_matrix = self.compute_similarity_matrix(h_rep_a_aux, h_rep_v_aux)
        similarity_matrix.fill_diagonal_(-1e3)
        hard_negatives_indices_audio = torch.argmax(similarity_matrix, dim=1)
        hard_negatives_indices_visual = torch.argmax(similarity_matrix, dim=0)


        hard_negatives_audio = h_rep_v_aux[hard_negatives_indices_audio]
        hard_negatives_visual = h_rep_a_aux[hard_negatives_indices_visual]

        postive_audio = h_rep_v_aux[torch.arange(h_rep_v_aux.size()[0])]
        postive_visual = h_rep_a_aux[torch.arange(h_rep_a_aux.size()[0])]

        # Process features through the auxiliary network
        processed_v = self.proj_aux3(F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(h_rep_v_aux)))))))
        processed_negative_audio = self.proj_aux3(
            F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(hard_negatives_audio)))))))
        processed_postive_audio = self.proj_aux3(
            F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(postive_audio)))))))

        processed_a = self.proj_aux3(F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(h_rep_a_aux)))))))
        processed_negative_video = self.proj_aux3(
            F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(hard_negatives_visual)))))))
        processed_postive_video = self.proj_aux3(
            F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(postive_visual)))))))

        # Classify features using the Matcher network
        predicted_match_v_a = self.matcher(processed_v, processed_negative_video)
        predicted_match_a_v = self.matcher(processed_a, processed_negative_audio)
        predicted_postive_v_a = self.matcher(processed_v, processed_postive_video)
        predicted_postive_a_v = self.matcher(processed_a, processed_postive_audio)

        # Compute the loss
        loss_v = F.binary_cross_entropy_with_logits(torch.cat([predicted_match_v_a, predicted_postive_v_a]), torch.cat([torch.zeros_like(predicted_match_v_a), torch.ones_like(predicted_postive_v_a)]))
        loss_a = F.binary_cross_entropy_with_logits(torch.cat([predicted_match_a_v, predicted_postive_a_v]), torch.cat([torch.zeros_like(predicted_match_a_v), torch.ones_like(predicted_postive_a_v)]))
        loss2 = (loss_v + loss_a) / 2

        # Cross Attention
        # Audio/Visual
        h_av = self.trans_a_with_v(proj_x_a.permute((1, 0, 2)), proj_x_v.permute((1, 0, 2)), proj_x_v.permute((1, 0, 2)))
        h_as = self.trans_a_mem(h_av)
        representation_audio = h_as[-1]

        # Visual/Audio
        h_va = self.trans_v_with_a(proj_x_v.permute((1, 0, 2)), proj_x_a.permute((1, 0, 2)), proj_x_a.permute((1, 0, 2)))
        h_vs = self.trans_v_mem(h_va)
        representation_visual = h_vs[-1]

        # Concatenating audiovisual representations
        av_h_rep = torch.cat([representation_audio, representation_visual], dim=1)

        # Main network output
        linear_hs_proj_av = self.proj2(
            F.dropout(F.relu(self.proj1(av_h_rep)), p=self.out_dropout, training=self.training))
        linear_hs_proj_av += av_h_rep
        output = self.out_layer(linear_hs_proj_av)

        return output, loss1, loss2
        