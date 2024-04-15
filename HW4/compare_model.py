import matplotlib.pyplot as plt

def read_fid_scores(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        fid_scores = [float(line.strip()) for line in lines]
    return fid_scores

dcgan_fid_scores = read_fid_scores('./result/dcgan_fid_scores.txt')
wgan_fid_scores = read_fid_scores('./result/wgan_fid_scores.txt')
acgan_fid_scores = read_fid_scores('./result/acgan_fid_scores.txt')

plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), dcgan_fid_scores, label='DCGAN FID Scores', marker='o')
plt.plot(range(1, 51), wgan_fid_scores, label='WGAN FID Scores', marker='x')
plt.plot(range(1, 51), acgan_fid_scores, label='ACGAN FID Scores', marker='s')

plt.legend()
plt.title('FID Scores over Epochs')
plt.xlabel('Epoch')
plt.ylabel('FID Score')
plt.grid(True)
plt.show()
