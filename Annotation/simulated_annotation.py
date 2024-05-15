from uitls import *

pkl_path = "/mnt/Jan/Py_PJ/OVAAL_FV/Annotation/Parking_garage/0_query/queries.pkl"
label_dir = "/mnt/Jan/Py_PJ/OVAAL_FV/Annotation/gt"
data = read_pkl(pkl_path)
print(data)
for p_img, dict_info in data.items():
    label_name = p_img.split("/")[-1].replace('frame', 'label_frame')
    label = cv2.imread(f"{label_dir}/{label_name}", -1)
    for points, _ in dict_info.items():
        data[p_img][points] = label[points]


save_pkl(data, pkl_path)
