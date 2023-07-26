from math import log
import os

DIRECTORY = os.path.realpath(os.path.dirname(__file__))

def parse_value(value):
    for op in [int, float, str]:
        try:
            return op(value)
        except:
            pass
    return None

def parse_line(line, sep=","):
    return [parse_value(j) for j in line.split(sep)]

def files_in_directory(directory):
    return [j for j in os.listdir(directory) if os.path.isfile(f"{directory}/{j}")]

def nelson_aalen(instances):
    ts = {}
    for inst in instances:
        t = inst.time
        d = inst.event

        if t not in ts:
            ts[t] = [0, 0]
        ts[t][d] += 1

    d = {0: 1 / (len(instances) + 1)}
    at_risk = len(instances)
    sum = 0
    for t in sorted(ts.keys()):
        left, died = ts[t]
        sum += died / at_risk
        at_risk -= died + left

        if died > 0:
            d[t] = sum

    keys = [*d.keys()]

    def f(x):
        try:
            return d[x]
        except:
            pass

        a = 0
        b = len(keys) - 1
        while a != b:
            mid = (a + b + 1) // 2
            if keys[mid] > x:
                b = mid - 1
            else:
                a = mid
        return d[keys[a]]

    return f

def kaplan_meier(instances):
    ts = {}
    for inst in instances:
        t = inst.time
        d = inst.event

        if t not in ts:
            ts[t] = [0, 0]
        ts[t][d] += 1

    d = {0: 1}
    at_risk = len(instances)
    for t in sorted(ts.keys()):
        left, died = ts[t]
        d[t] = died / at_risk
        at_risk -= died + left

    keys = [*d.keys()]
    def f(x):
        try:
            return d[x]
        except:
            pass

        a = 0
        b = len(keys) - 1
        while a != b:
            mid = (a + b + 1) // 2
            if keys[mid] > x:
                b = mid - 1
            else:
                a = mid
        return d[keys[a]]

    return f

class Instance:
    def __init__(self, feats):
        self.time = feats.pop("time")
        self.event = int(float(feats.pop("event")) > 0.5)
        self.feats = feats

    def __repr__(self):
        return f"<t = {self.time}, d = {self.event}, feats = {self.feats}>"

def calculate_theta(events, hazards):
    if len(events) == 0:
        return -1

    numerator = sum(events)
    denominator = sum(hazards)

    theta = 0
    if denominator > 0:
        theta = numerator / denominator

    return theta

class Tree:
    def __init__(self, criterium, tree_0, tree_1, instances=None):
        self.criterium = criterium
        self.trees = [tree_0, tree_1] if tree_0 else []
        self.instances = instances or []

        self.theta = None
        self.distribution = None
        self.error = None

        if instances:
            self.calculate_label()
            self.calculate_error()
    
    def size(self):
        if not self.trees:
            return 0
        return self.trees[0].size() + 1 + self.trees[1].size()

    def traverse(self, instance, store=False):
        if store:
            self.instances.append(instance)

        if not self.trees:
            return (self.theta, self.distribution)

        try:
            if not self.criterium(instance.feats):
                return self.trees[0].traverse(instance, store)
            else:
                return self.trees[1].traverse(instance, store)
        except:
            print(self)
            print(instance.feats)

    def calculate_label(self):
        events = [inst.event for inst in self.instances]
        hazards = [Tree.hazard_function(inst.time) for inst in self.instances]

        self.theta = calculate_theta(events, hazards)
        self.distribution = kaplan_meier(self.instances)

        return (self.theta, self.distribution)

    def calculate_error(self):
        if self.trees:
            self.error = 0
            for i in range(2):
                if self.trees[i].error == None:
                    self.trees[i].error = self.trees[i].calculate_error()
                self.error += self.trees[i].error
            return self.error

        if self.theta == None or self.distribution == None:
            self.theta, self.distribution = self.calculate_label()

        self.error = 0
        for inst in self.instances:
            event = inst.event
            hazard = Tree.hazard_function(inst.time)

            instance_error = hazard * self.theta
            if event:
                if self.theta == 0:
                    return 0
                if self.theta < -0.5:
                    continue
                instance_error -= log(hazard) + log(self.theta) + 1
            self.error += instance_error
        self.error = max(0, self.error)

        return self.error

    def get_leaves(self):
        if self.trees:
            return self.trees[0].get_leaves() + self.trees[1].get_leaves()
        return [self]

    def clear_instances(self):
        self.error = None
        self.instances = []
        for child in self.trees:
            child.clear_instances()

    def to_string(self, path):
        trace = "".join(["\033[30m:\033[0m     ", "║     ", "╚═\033[31;1mX\033[0m═══", "╠═\033[32;1mV\033[0m═══"][path[j] + 2 * (j == len(path) - 1)] for j in range(len(path)))

        r = []
        if not self.trees:
            theta_repr = "???"
            if self.theta != None:
                theta_repr = f"{self.theta:.4f}"
            r.append(f"{trace}\033[34;1m(\033[36m{theta_repr}\033[34m)\033[0m ")
        else:
            r.append(f"{trace}\033[43;38;2;0;0;0m X \033[0m ")
            path.append(1)
            r.append(self.trees[1].to_string(path))
            path[-1] = 0
            r.append(self.trees[0].to_string(path))
            path.pop()

        if self.instances:
            if self.trees:
                r[0] += "\033[38;5;232m"
            else:
                r[0] += "\033[38;5;248m"
            r[0] += f"<error = {self.error or 0:.4f}, avg-error = {(self.error or 0) / len(self.instances):.4f}, {sum(inst.event for inst in self.instances)} / {len(self.instances)} died>\033[0m"
        r[0] += "\n"
        return "".join(r)

    def __repr__(self):
        return self.to_string([]).strip()

def get_feature_meanings(filename):
    if len(filename.split("/")):
        filename = "_".join(filename.split("/")[1].split("_")[:-1])

    f = open(f"{DIRECTORY}/metadata/{filename}_features.txt")
    feature_meanings = f.read().split("\n")
    f.close()

    return feature_meanings

def read_dataset(filename):
    f = open(filename)
    lines = f.read().strip().split("\n")
    f.close()

    keys, lines = lines[0].split(","), lines[1:]
    
    instances = []
    for values in [parse_line(line, sep=",") for line in lines]:
        inst = Instance({key: value for key, value in zip(keys, values)})
        instances.append(inst)

    return instances

def parse_tree(d):
    if len(d) == 1:
        tree = Tree(-1, None, None)
        tree.theta = d[0]
        return tree
    else:
        feat = d[0]
        tree_0 = parse_tree(d[1])
        tree_1 = parse_tree(d[2])
        return Tree(feat, tree_0, tree_1)

def read_tree(filename):
    f = open(filename)
    dataset_filename, d = f.read().strip().split("\n")
    f.close()

    null = None
    flat_tree = eval(d)
    return parse_tree(flat_tree), dataset_filename

def fill_tree(tree, dataset_filename):
    instances = read_dataset(dataset_filename)

    Tree.hazard_function = nelson_aalen(instances)

    for inst in instances:
        tree.traverse(inst, True)
    tree.calculate_error()

    return instances
