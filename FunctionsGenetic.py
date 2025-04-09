import random 
def greedy_random_walk(graph, start, target, max_steps=1000):
    path = [(start, 0)]  # Start with the initial node and zero cost.
    current = start
    
    for _ in range(max_steps):
        if current == target:
            return path
        
        # Retrieve neighbors of the current node.
        neighbors = graph.edges(current)
        if not neighbors:
            break  # No path if there are no outgoing edges.
        
        # Optional: Filter out neighbors to avoid cycles.
        valid_neighbors = [(nbr, graph.edges[(itself, nbr)]["weight"]) for itself, nbr in neighbors if nbr not in path]
        if not valid_neighbors:
            # If all neighbors lead to a cycle, allow revisiting.
            valid_neighbors = [(nbr, graph.edges[(itself, nbr)]["weight"]) for itself, nbr in neighbors]
        # Calculate weights: lower cost edges should have higher probability.


        # We use the inverse of the cost (handling zero cost to avoid division by zero).
        weights = []
        for nbr, cost in valid_neighbors:
            adjusted_cost = cost if cost > 0 else 1e-6  # Avoid division by zero.
            weights.append(1.0 / adjusted_cost)
        
        # Randomly choose the next node based on the computed weights.
        next_node = random.choices(
            [nbr for nbr, _ in valid_neighbors],
            weights=weights,
            k=1
        )[0]

        path.append((next_node, graph.edges[(current, next_node)]["weight"] + path[-1][1]))
        current = next_node
    
    # If target wasn't reached within max_steps, return None.
    return None


def rouletteWheel(chrom,k):
    chrom=sorted(chrom,key=lambda x: x[-1][1])
    chrom=chrom[:k]
    total=sum([1/x[-1][1] for x in chrom])
    probabilities=[(1/x[-1][1])/total for x in chrom]
    return random.choices(chrom,weights=probabilities,k=1)[0]



def crossover(p1,p2, G):
    p1Set=set([x[0] for x in p1])
    # print(p1Set)
    childs=[]
    for i, node in zip(range(1,len(p2)), p2[1:]):
        if node[0] in p1Set:
            index=-1.5
            for j in range(1,len(p1)):
                if(node[0]==p1[j][0]):
                    index=j
                    break
            # print(index,i)
            if(index==-1.5):
                continue
            child1, child2 =p1[:index]+p2[i:], p2[:i]+p1[index:]

            for j in range(index,index + len(p2[i:])):
                child1[j]=(child1[j][0], G.edges[(child1[j-1][0],child1[j][0])]["weight"] + child1[j-1][-1])

            for j in range(i, i + len(p1[index:])):
                child2[j]=(child2[j][0],G.edges[(child2[j-1][0],child2[j][0])]["weight"] + child2[j-1][-1])
            childs.append(child1)
            childs.append(child2)
    return childs
# childs=crossover(p1,p2)


# s,e=3,5
def mutation(childs, index,s,e, G):
    try:
        mutation=greedy_random_walk(G,childs[index][s][0],childs[index][e][0])


        for i in range(1,len(mutation)):
            childs[index].insert(s+i,mutation[i])
            childs[index][s+i]=(childs[index][s+i][0],G.edges[(childs[index][s+i-1][0],childs[index][s+i][0])]["weight"] + childs[index][s+i-1][-1])

        for i in range(s+len(mutation),len(childs[2])):
            childs[index][i]=(childs[index][i][0], G.edges[(childs[index][i-1][0],childs[index][i][0])]["weight"] + childs[index][i-1][-1])

    except:
        pass
    # print(len(childs[index]), len(childs))
    return childs



def genetic_algorithm(G ,start, end, sampleSize=100, parentCount=10, generations=100):
    chromosomes = []
    for _ in range(sampleSize):
        path = greedy_random_walk(G, start, end)# Generate initial population using a greedy random walk
        chromosomes.append(path)
    generation = []
    # print("chromosomes",len(chromosomes))
    for _ in range(parentCount):
        p1 = rouletteWheel(chromosomes, int(sampleSize * 0.4)) 
        p2 = rouletteWheel(chromosomes, int(sampleSize * 0.4)) # Select two parents using the roulette wheel selection method
        childs = crossover(p1, p2, G)
        # Introduces random mutations in some offspring with a low probability (3%)
        # print("child Count",len(childs))
        for i in range(len(childs)):
            r= random.random()
            # print("random ",r)
            if r < 0.03:
                s = random.randint(1, len(childs[i]) // 2)
                e = random.randint(s + 1, len(childs[i]) - 1)
                if(e-s>2):
                    # print("mutation")
                    childs[i] = mutation(childs, i, s, e, G)
        for child in childs:
            # print("child",child[-1][1])
            if(type(child[-1][1])!=int):
                childs.remove(child)
                # print(len(childs))
        childs = sorted(childs, key=lambda x: x[-1][1]) # Sort offspring based on the cost

        childs = childs[:len(childs) // 2]
        for child in childs:
            generation.append(child) # Add surviving children to the next generation
        # print(len(generation))

    # Find the best chromosome (shortest or most optimal path)
    best = ("", float("inf"))
    index = -1.5  # Initialize with an invalid index
    for i, chrom in zip(range(len(generation)), generation):
        if chrom[-1][1] < best[1]:
            best = chrom[-1]
            index = i 
    return generation[index]  # Return the best path

# # Execute the genetic algorithm and get the best route

# start = 'Chicago, IL'
# end = 'New York City, NY (Metropolitan Area)'
# # Start:  Chicago, IL  End:  New York City, NY (Metropolitan Area)
# import time

# start_time = time.time()
# route = genetic_algorithm(start, end, sampleSize=50, parentCount=10, generations=100)
# print("--- %s seconds Genetic Algorithm---" % (time.time() - start_time))

# route[-1][-1]