import torch

# checkpoint 불러오기
checkpoint = torch.load('ocr_pretrained.pth.tar', map_location='cpu')

# state_dict 가져오기 (없으면 checkpoint가 바로 state_dict일 수도)
state_dict = checkpoint.get('state_dict', checkpoint)

# 키들 출력해보기 (파라미터 이름들)
for key in state_dict.keys():
    print(key)
