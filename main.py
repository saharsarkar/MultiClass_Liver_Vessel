# -*- coding: utf-8 -*-
"""
@author: S.Sarkar

"""

# Define Model
model = UneTR(img_dim,
               in_channels=1,
               base_filter=16,
               class_num=3,
               patch_size=16,
               embedding_dim=768,
               block_num=12,
               head_num=12,
               mlp_dim=3072,
               z_idx_list=[3, 6, 9, 12]).to(device)

loss_function = DiceCELoss(softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

epochs = 700
csv_filename = '/path/model_history_log.csv'
history = train(model,
                train_dataloader,
                validation_dataloader,
                optimizer,
                loss_function,
                epochs,
                device,
                csv_filename)