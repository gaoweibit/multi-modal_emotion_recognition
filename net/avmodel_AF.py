import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder
import torchvision.models as models


class Matcher(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Matcher, self).__init__()
        self.linear1 = nn.Linear(input_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


class AVmodel(nn.Module):
    def __init__(self, model_args):
        super(AVmodel, self).__init__()

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
        combined_dim = 2 * self.d_v

        self.conv_1d_a = nn.Conv1d(self.a_dim, self.d_v, kernel_size=1, padding=0, bias=False)

        self.a_mem = self.transformer_arch(self_type='audio_self')
        self.v_mem = self.transformer_arch(self_type='visual_self')
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

        self.mapping_layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, model_args.v_dim)
        )

        # Add Matcher network
        self.matcher = Matcher(self.d_v, 256)


    def transformer_arch(self, self_type='audio/visual', scalar=False, pos_emb=False):
        if self_type == 'visual/audio':
            embed_dim, attn_dropout = self.d_v, 0
        elif self_type == 'audio/visual':
            embed_dim, attn_dropout = self.d_v, 0
        elif self_type == 'audio_self':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        elif self_type == 'visual_self':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
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
        audio_features = F.normalize(audio_features, dim=-1)
        visual_features = F.normalize(visual_features, dim=-1)

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

        # Audio/Visual
        h_av = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = self.trans_a_mem(h_av)
        representation_audio = h_as[-1]

        # Visual/Audio
        h_va = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = self.trans_v_mem(h_va)
        representation_visual = h_vs[-1]

        # Concatenating audiovisual representations
        av_h_rep = torch.cat([representation_audio, representation_visual], dim=1)

        # Auxiliary audio network
        h_a1 = self.a_mem(proj_x_a)
        h_a2 = self.a_mem(h_a1)
        h_a3 = self.a_mem(h_a2)
        h_rep_a_aux = h_a3[-1]

        # Auxiliary visual network
        h_v1 = self.v_mem(proj_x_v)
        h_v2 = self.v_mem(h_v1)
        h_v3 = self.v_mem(h_v2)
        h_rep_v_aux = h_v3[-1]


        # Calculate hard negatives.
        similarity_matrix = self.compute_similarity_matrix(h_rep_a_aux, h_rep_v_aux)
        similarity_matrix.fill_diagonal_(-1e3)
        hard_negatives_indices_audio = torch.argmax(similarity_matrix, dim=1)
        hard_negatives_indices_visual = torch.argmax(similarity_matrix, dim=0)


        hard_negatives_audio = h_rep_v_aux[hard_negatives_indices_audio]
        hard_negatives_visual = h_rep_a_aux[hard_negatives_indices_visual]

        # Process features through the auxiliary network
        processed_v = self.proj_aux3(F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(h_rep_v_aux)))))))
        processed_negative_audio = self.proj_aux3(
            F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(hard_negatives_audio)))))))

        processed_a = self.proj_aux3(F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(h_rep_a_aux)))))))
        processed_negative_video = self.proj_aux3(
            F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(hard_negatives_visual)))))))

        # Classify features using the Matcher network
        predicted_match_v_a = self.matcher(processed_v, processed_negative_video)
        predicted_match_a_v = self.matcher(processed_a, processed_negative_audio)

        # Compute the loss
        loss_v = F.binary_cross_entropy_with_logits(predicted_match_v_a, torch.zeros_like(predicted_match_v_a))
        loss_a = F.binary_cross_entropy_with_logits(predicted_match_a_v, torch.zeros_like(predicted_match_a_v))

        loss = (loss_v + loss_a) / 2

        # Main network output
        linear_hs_proj_av = self.proj2(
            F.dropout(F.relu(self.proj1(av_h_rep)), p=self.out_dropout, training=self.training))
        linear_hs_proj_av += av_h_rep
        output = self.out_layer(linear_hs_proj_av)

        return output, loss
