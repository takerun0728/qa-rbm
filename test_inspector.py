from dwave.system import DWaveSampler, FixedEmbeddingComposite, EmbeddingComposite
import dwave.inspector
import os

h = {}
J = {}
for i in range(512):
    J[(i*8, i*8+4)] = 1
    J[(i*8+1, i*8+5)] = -1
    J[(i*8+2, i*8+6)] = 1
    J[(i*8+3, i*8+7)] = -1
del J[(42, 46)]
del J[(43, 47)]
del J[(520, 524)]
del J[(544, 548)]
del J[(1723, 1727)]
del J[(1731, 1735)]
del J[(1800, 1804)]

embedding = {}
for i in range(2048):
    h[i] = 0.1
    embedding[i] = [i] 
del embedding[42]
del embedding[43]
del embedding[46]
del embedding[47]
del embedding[520]
del embedding[524]
del embedding[544]
del embedding[548]
del embedding[1723]
del embedding[1727]
del embedding[1731]
del embedding[1735]
del embedding[1800]
del embedding[1804]
sampler = DWaveSampler(solver='DW_2000Q_6', token=os.environ.get("dwave_token"))
embedded_sampler = FixedEmbeddingComposite(sampler, embedding=embedding)

for i in range(2048):
    if not(i in sampler.nodelist):
        print(i)

response = embedded_sampler.sample_ising(h, J, num_reads=10)
dwave.inspector.show(response) 