file_path = "citeseer"

network = './dataset/' + file_path + '/network.dat'
community = './dataset/' + file_path + '/community.dat'

output_network = './dataset/' + file_path + '/network1.dat'
output_community = './dataset/' + file_path + '/community1.dat'
# Load network and convert node id -> node id -1

with open(network, 'r') as f:
    lines = f.readlines()
    with open(output_network, 'w') as f:
        for line in lines:
            line = line.strip().split()
            f.write(str(int(line[0])-1) + '\t' + str(int(line[1])-1) + '\n')


# Load community and convert node id -> node id -1
with open(community, 'r') as f:
    lines = f.readlines()
    with open(output_community, 'w') as f:
        for line in lines:
            line = line.strip().split()
            f.write(str(int(line[0])-1) + '\t' + str(int(line[1])) + '\n')
