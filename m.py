a = [1,2,3]
b = ['a','b','c']

# for i,j in zip(a,b):
#     print(i,j)
# idx = [1,2,1]
# my_list = ['apple', 'banana', 'orange']
# elements = [my_list[i-1] for i in idx]
# print(elements)
p = '/home/cognitica_ai_user/NPS/Github/Confusion_Matrix_for_Objecti_Detection_Models/sample_data/pred/blossom_end_rot_0010.txt'
g = 'sample_data/gt/blossom_end_rot_0010.txt'
p_s = (p.split('/'))[-1]
g_s = (g.split('/'))[-1]
if p_s==g_s:
    print('same')
print(p_s,g_s)