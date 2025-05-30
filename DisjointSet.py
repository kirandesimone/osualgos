class DisjointSet:
    def __init__(self, n: int) -> None:
        self.component_sizes = [1] * n
        self.components = [0] * n
        self.num_of_components = n

        for i in range(n):
            self.components[i] = i

    def find(self, v: int) -> int:
        root = v
        while root != self.components[root]:
            root = self.components[root]

        while v != root:
            old_root = self.components[v]
            self.components[v] = root
            v = old_root

        return root

    def union(self, v1: int, v2: int) -> bool:
        r1 = self.find(v1) 
        r2 = self.find(v2)

        if r1 == r2:
            return False

        if self.component_sizes[r1] > self.component_sizes[r2]:
            self.components[r2] = r1
            self.component_sizes[r1] += self.component_sizes[r2]
        elif self.component_sizes[r1] < self.component_sizes[r2]:
            self.components[r1] = r2
            self.component_sizes[r2] += self.component_sizes[r1]
        else:
            self.components[r2] = r1
            self.component_sizes[r1] += self.component_sizes[r2]

        self.num_of_components -= 1 

        return True

    def numOfComponents(self) -> int:
        return self.num_of_components 
