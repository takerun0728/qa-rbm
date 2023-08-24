from dwave.system import DWaveSampler, FixedEmbeddingComposite, EmbeddingComposite
import dwave.inspector
import os

token = os.environ.get("dwave_token")

START_HOR = 60
START_VER = 3000
DEAD_NODES = [616, 727, 742, 771, 1261, 1450, 1705, 1905, 1936, 1996, 2025, 3042, 3052, 3069, 3127, 3204, 3413, 3422, 3597, 3852, 4442, 4666, 4949, 5207, 5460, 5685]
DEAD_LINKS = [(4270, 2032), (5118, 641), (4832, 4833)]

def get_var_node(r, c, grp, idx):
    return r * 180 + grp * 60 + c + 3000 + 15 * idx

def get_hor_node(r, c, grp, idx):
    return c * 180 + (2 - grp) * 60 + r + 60 + 15 * idx

h = {}
J = {}
for r in range(15):
    for c in range(15):
        for k in range(3):
            for var_idx in range(4):
                J[get_var_node(r, c, k, var_idx), get_hor_node(r, c, k, 0)] = 1
                J[get_var_node(r, c, k, var_idx), get_hor_node(r, c, k, 1)] = 0.5
                J[get_var_node(r, c, k, var_idx), get_hor_node(r, c, k, 2)] = -0.5
                J[get_var_node(r, c, k, var_idx), get_hor_node(r, c, k, 3)] = -1

for r in range(15):
    for c1, c2 in zip(range(0, 14), range(1, 15)):
        for k in range(3):
            J[get_var_node(r, c1, k, 0), get_var_node(r, c2, k, 0)] = 1
            J[get_var_node(r, c1, k, 1), get_var_node(r, c2, k, 1)] = 0.5
            J[get_var_node(r, c1, k, 2), get_var_node(r, c2, k, 2)] = -0.5
            J[get_var_node(r, c1, k, 3), get_var_node(r, c2, k, 3)] = -1

for r1, r2 in zip(range(0, 14), range(1, 15)):
    for c in range(15):
        for k in range(3):
            J[get_hor_node(r1, c, k, 0), get_hor_node(r2, c, k, 0)] = 1
            J[get_hor_node(r1, c, k, 1), get_hor_node(r2, c, k, 1)] = 0.5
            J[get_hor_node(r1, c, k, 2), get_hor_node(r2, c, k, 2)] = -0.5
            J[get_hor_node(r1, c, k, 3), get_hor_node(r2, c, k, 3)] = -1

dead_links = DEAD_LINKS.copy()
for key in J.keys():
    for dead in DEAD_NODES:
        if dead in key:
            dead_links.append(key)
for dead_link in dead_links:
    del J[dead_link]

embedding = {}
for i in range(60, 2760):
    embedding[i] = [i] 
for i in range(3000, 5700):
    embedding[i] = [i]
for dead in DEAD_NODES:
    del embedding[dead]

sampler = DWaveSampler(solver='Advantage_system6.2', token=os.environ.get("dwave_token"))
embedded_sampler = FixedEmbeddingComposite(sampler, embedding=embedding)

for i in range(60, 2760):
    if not(i in sampler.nodelist):
        print(i)
for i in range(3000, 5700):
    if not(i in sampler.nodelist):
        print(i)

response = embedded_sampler.sample_ising(h, J, num_reads=10)
dwave.inspector.show(response) 