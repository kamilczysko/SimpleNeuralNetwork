import neural_network as nn
import numpy as np
import prep_data_loader as data
from PIL import Image

# x = np.array([[1,0],[0,1],[0,0],[1,1]])
# y = np.array([[0],[0],[1],[1]])
x = data.learn_data_array
y = data.learn_target_array
ans_arr = data.answer_array

brain = nn.NeuralNetwork(784, 65, 6)

print(y[1],' ----- y1')
print(x[1],' ----- x1')

def teach():
    for i in range(1):
        for j in range(len(x)):
            brain.train([x[j]], y[j])
    print('1 epoch done')
print(ans_arr)

def feedforward(data):
    answer = brain.feedforward(data)
    index_of_max = np.argmax(np.array(answer))
    ans = ''
    max_percent = np.max(answer)

    for i in ans_arr:
       if index_of_max in i:
           ans = i[0]

    print('guess: ')
    print(answer)
    print(ans,' - ',max_percent)

    # img = np.reshape(data * 255, newshape=(28, 28))
    # pic = Image.fromarray(img)
    # pic.show()
    print(answer)


# print('before')
# feedforward(x[2])
# teach()
# print('after')
# feedforward(x[0])
# feedforward(x[1])
# feedforward(x[2])
# feedforward(x[3])

#
# print(brain.weights_HO,' --- ',brain.weights_IH)