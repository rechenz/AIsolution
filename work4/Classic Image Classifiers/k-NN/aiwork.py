
# def find_the_k():
#     total_num = 10000.0
#     max_accuracy = 0.0
#     best_k = None

#     for k in range(1, 11):
#         total_correct = 0.0
#         print(f"Testing k={k}")

#         for check_id in range(1, 6):
#             print(f"Testing batch {check_id}/5")
#             batch_correct = 0.0

#             for i in range(10000):
#                 test_sample = data[check_id-1][i]
#                 check_arr = np.zeros(10)
#                 distance_list = []

#                 # 计算与其他所有batch的距离
#                 for test_id in range(1, 6):
#                     if check_id == test_id:
#                         continue

#                     train_data = data[test_id-1]
#                     # 向量化计算距离
#                     distances = np.sqrt(
#                         np.sum(np.square(test_sample - train_data), axis=(1, 2, 3)))
#                     distance_list.extend(distances)

#                 # 获取k个最近邻
#                 distance = np.array(distance_list)
#                 nearest_indices = np.argpartition(distance, k)[:k]

#                 # 统计标签
#                 for idx in nearest_indices:
#                     batch_idx = idx // 10000
#                     sample_idx = idx % 10000
#                     check_arr[datalabel[batch_idx][sample_idx]] += 1

#                 # 预测标签
#                 predicted_label = np.argmax(check_arr)
#                 if predicted_label == datalabel[check_id-1][i]:
#                     batch_correct += 1

#             total_correct += batch_correct

#         # 计算准确率
#         accuracy = total_correct / (total_num * 5)
#         print(f"Accuracy for k={k}: {accuracy}")

#         if accuracy > max_accuracy:
#             max_accuracy = accuracy
#             best_k = k

#     return best_k
