"""
Code for compressing and decompressing using Huffman compression.
"""

#Source:
##http://homes.sice.indiana.edu/yye/lab/teaching/spring2014-C343/huffman.php
##https://www.siggraph.org/education/materials/HyperGraph/video/mpeg/mpegfaq/huffman_tutorial.html
##https://betterexplained.com/articles/understanding-big-and-little-endian-byte-order/

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict(int,int)

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    dictionary = {}
    for key in text:
        if not key in dictionary:
            dictionary[key] = 1
        else:
            dictionary[key] += 1
    return dictionary

def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    #Make sure not to turn it into a node if it already is a node...
    #You're currently doing that. You have to check
    if len(freq_dict) == 1:
        return HuffmanNode(None,
                           HuffmanNode(sorted(freq_dict.keys())[0]), None)
    reverse = reverse_dict(freq_dict)
    while len(get_values(reverse)) > 2:
        sorted_freq = sorted(reverse.keys())
        freq = sorted_freq[0]
        if len(reverse[freq]) > 1:
            tree = HuffmanNode(None, reverse[freq][0],
                               reverse[freq][1])            
            if freq*2 not in reverse:
                reverse[freq*2] = [tree]
            else:
                reverse[freq*2].append(tree)
            if len(reverse[freq]) > 2:
                reverse[freq].pop(0)
                reverse[freq].pop(0)
            else:
                del reverse[freq]
        else:
            freq_2 = sorted_freq[1]
            tree = HuffmanNode(None, reverse[freq][0],
                               reverse[freq_2][0])             
            if freq + freq_2 not in reverse:
                reverse[freq + freq_2] = [tree]
            else:
                reverse[freq + freq_2].append(tree)
            del reverse[freq]
            if len(reverse[freq_2]) > 1:
                reverse[freq_2].pop(0)
            else:
                del reverse[freq_2]
    return HuffmanNode(None,
                       get_values(reverse)[0],
                       get_values(reverse)[1])

def get_values(d):
    '''return all values in dictionary d'''
    final_list = []
    for freq in d:
        for elem in d[freq]:
            final_list.append(elem)
    return final_list
                       
        

def sort_elem(reverse, sorted_freq):
    '''return the elements in a list sorted by their frequency'''
    final_list = []
    for freq in sorted_freq:
        for element in reverse[freq]:
            if element not in final_list:
                final_list.append(element)
    return final_list
        
    
def reverse_dict(freq_dict):
    '''return the reverse of freq_dict where each value is mapped
    to its keys'''
    reverse = {}
    for key in freq_dict:
        if not freq_dict[key] in reverse:
            reverse[freq_dict[key]] = [HuffmanNode(key)]
        else:
            reverse[freq_dict[key]].append(HuffmanNode(key))
    return reverse
            

def get_codes(tree):
    """ Return a dict mapping symbols from Huffman tree to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree = HuffmanNode(None, HuffmanNode(3), None)
    >>> d = get_codes(tree)
    >>> d
    {3: "0"}
    """
    d = {}
    code_list = bin_codes(tree)
    i = 0
    while i < len(code_list) :
        d[code_list[i]] = code_list[i+1]
        i += 2
    return d

def bin_codes(tree, code = ''):
    '''Return a list with tree symbol followed by its code'''
    code_list = []
    if tree == None:
        pass
    
    elif tree.is_leaf():
        return [tree.symbol, code]
    
    elif tree.right == None:
        return bin_codes(tree.left, code + '0')
    
    elif tree.left == None:
        return bin_codes(tree.right, code + '1')
    
    else:
        return bin_codes(tree.left, code + '0') + \
               bin_codes(tree.right, code + '1')
        

def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
        
    node_list = post_order(tree)
    i = 0
    for t in node_list:
        t.number = i
        i += 1
    
def post_order(tree):
    '''Return a postorder list of all trees in tree'''
    if tree.left == None:
        return []
    else:
        left = post_order(tree.left)
        
    if tree.right == None:
        return []
    else:
        right = post_order(tree.right)
    return left + right + [tree]

def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    total_freq = 0
    total_bits = 0
    codes = get_codes(tree)
    for symbol in freq_dict:
        total_freq += freq_dict[symbol]
        total_bits += len(codes[symbol]) * freq_dict[symbol]
    return total_bits / total_freq


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mapping from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    >>> text = bytes([2, 2, 1, 1, 0, 2, 1])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result] == ['11111010', '01110000']
    True
    >>> text = bytes([1, 2, 1, 0,10,214,14,43,63,77,45,32,246,11])
    >>> freq = make_freq_dict(text)
    >>> t = huffman_tree(freq)
    >>> codes = get_codes(t)
    >>> result = generate_compressed(text, codes)
    >>> [byte_to_bits(byte) for byte in result]
    ['01001100', '10011110', '00100110', '10101111', \
    '00110111', '10111100', '00010000']
    >>> [byte_to_bits(byte) for byte in text]
    ['00000001', '00000010', '00000001', '00000000', '00001010', '11010110',\
    '00001110', '00101011', '00111111', '01001101', '00101101', '00100000', \
    '11110110', '00001011']
    """
    
    #Compressing first solution
    s = ''
    for byte in text:
        code = codes[byte]
        s += code
    lst = []
    for i in range(0, len(s), 8):
        lst.append(bits_to_byte(s[i:i+8]))
    return bytes(lst)
    #
    
#Recursive way works but takes too long  
##    s = ''
##    for byte in text:
##        code = codes[byte]
##        s += code
##        #Use helper function to build list for the string of the codes
##    def make_list(s):
##        if len(s) < 9:
##            return [bits_to_byte(s)]
##        return [bits_to_byte(s[0:8])] + make_list(s[8:])
##
##    return bytes(make_list(s))

def tree_to_bytes(tree):
    """ Return a bytes representation of the Huffman tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    tree_list = post_order(tree)
    lst = []
    for t in tree_list:
        #Left subtree
        if t.left.is_leaf():
            lst.append(0)
            lst.append(t.left.symbol)
        else:
            lst.append(1)
            lst.append(t.left.number)
        #Right subtree 
        if t.right.is_leaf():
            lst.append(0)
            lst.append(t.right.symbol)
        else:
            lst.append(1)
            lst.append(t.right.number)
    return bytes(lst)

def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file to compress
    @param str out_file: output file to store compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in node_lst.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    root = node_lst[root_index]
    if root.l_type == 0:
        left =  HuffmanNode(root.l_data)
    elif root.l_type == 1:
        left = generate_tree_general(node_lst,
                                     root.l_data)
    if root.r_type == 0:
        right = HuffmanNode(root.r_data)
    elif root.r_type == 1:
        right = generate_tree_general(node_lst,
                                     root.r_data)

    return HuffmanNode(None , left, right)


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that node_lst represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2) == HuffmanNode\
    (None, HuffmanNode(None, HuffmanNode(5, None, None), \
    HuffmanNode(7, None, None)), \
    HuffmanNode(None, HuffmanNode(10, None, None), \
    HuffmanNode(12, None, None)))
    True
    
    >>> lst = [ReadNode(0, 3, 0, 4), ReadNode(1, 0, 0, 1)]
    >>> generate_tree_postorder(lst, 1)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(3, None, None),\
    HuffmanNode(4, None, None)), HuffmanNode(1, None, None))
    
    >>> lst = [ReadNode(0, None, 0, 2)]
    >>> generate_tree_postorder(lst, 0)
    HuffmanNode(None, HuffmanNode(None, None, None), HuffmanNode(2, None, None))

    >>> lst = [ReadNode(0, 2, 0, 3), ReadNode(1, 1, 0, 4)]
    >>> generate_tree_postorder(lst, 1)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(2,None, None), \
    HuffmanNode(3, None, None)),HuffmanNode(4, None, None))
    """
    root = node_lst[root_index]
    if root.r_type == 0:
        right = HuffmanNode(root.r_data)
    elif root.r_type == 1: #Right subtree is a tree
        right = generate_tree_postorder(node_lst, root_index -1)

    if root.l_type == 0:
        left = HuffmanNode(root.l_data)
    elif root.l_type == 1: #Left subtree is a tree
        #The left subtree will be before the rightmost leaf in postorder
        root_index -= len(post_order(right))
        left = generate_tree_postorder\
               (node_lst, root_index - 1)
    return HuffmanNode(None, left, right)

#Helper for generating uncompresesd:

def reverse_code(codes):
    '''Return the dictionary from codes to element'''
    reverse = {}
    for code in codes:
        reverse[codes[code]] = code
    return reverse

def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: number of bytes to decompress from text.
    @rtype: bytes

    >>> with open("book.txt", "rb") as f1:
            text = f1.read()
    >>> freq = make_freq_dict(text)
    >>> tree = huffman_tree(freq)
    >>> codes = get_codes(tree)
    >>> number_nodes(tree)
    >>> size = len(text)
    >>> compressed = generate_compressed(text, codes)
    >>> uncompressed = generate_uncompressed(tree, compressed, size)
    >>> text == uncompressed
    True
    """
    codes = get_codes(tree)
    reverse = reverse_code(codes)
    lst = []
    t = ''
    for i in text:
        t += byte_to_bits(i)
        
    i = 0 #Start string pointer
    j = 1 #End string pointer

    while len(lst) != size:
        if t[i:j] in reverse:
            lst.append(reverse[t[i:j]])
            i = j #The beginning of new code is where last code ended
            j = i+1 #New code ending is one after i 
        else:
            j += 1
    return bytes(lst)

    
def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
##        tree = generate_tree_postorder(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31

    >>> left = HuffmanNode(2)
    >>> right = HuffmanNode(None, HuffmanNode(3), HuffmanNode(4))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {2: 5, 3: 7, 4: 10}
    >>> avg_length(tree, freq)
    1.7727272727272727
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    1.5454545454545454

    >>> left = HuffmanNode(None, HuffmanNode(None, HuffmanNode(20),\
    HuffmanNode(3)), HuffmanNode(1))
    >>> right = HuffmanNode(None, HuffmanNode(None, HuffmanNode(8),\
    HuffmanNode(7)), HuffmanNode(2))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {20:1, 1:3, 3:80, 8:4, 7:10, 2:100}
    >>> avg_length(tree, freq)
    2.4797979797979797
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.090909090909091

    >>> left = HuffmanNode(None, HuffmanNode(None, HuffmanNode(None,\
    HuffmanNode(1), HuffmanNode(7)), HuffmanNode(8)), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(5), HuffmanNode(None,\
    HuffmanNode(4), HuffmanNode(3)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {1:20, 7:30, 8:15, 2:1, 5:3, 4:100, 3:7}
    >>> avg_length(tree, freq)
    3.2613636363636362
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.284090909090909
    """
    #Algorithm: If tree has two leaves, don't do anything
    reverse_freq = reverse_dict(freq_dict)
    sorted_freq = sorted(reverse_freq.keys())
    depth_order = []
    for i in sorted_freq:
        depth_order += reverse_freq[i]
    leaf_depths_dic = leaf_depths(tree, 0, {})
    final_dict = {} #Keeps track of which nodes will be in which depth
    for i in sorted(leaf_depths_dic.keys()):
        for j in range(leaf_depths_dic[i]):
            if i not in final_dict:
                final_dict[i] = [depth_order.pop()]
            else:
                final_dict[i].append(depth_order.pop())
    change_leaves(tree, final_dict, 1)
        
def change_leaves(tree, depth_dict, depth):
    '''change all leaves to make tree optimal'''
    if tree.left.is_leaf():
        tree.left = depth_dict[depth].pop()
    else:
        change_leaves(tree.left, depth_dict, depth + 1)

    if tree.right.is_leaf():
        tree.right = depth_dict[depth].pop()
    else:
        change_leaves(tree.right, depth_dict, depth + 1)

def leaf_depths(tree, depth, d):
    '''Traverse through tree and get number
    of leaves at each depth'''
    if tree.is_leaf():
        if depth not in d:  
            d[depth] = 1
        else:
            d[depth] += 1
    elif tree.left == None:
        leaf_depths(tree.right, depth + 1, d)
    elif tree.right == None:
        leaf_depths(tree.left, depth + 1, d)
    else:
        leaf_depths(tree.left, depth + 1, d)
        leaf_depths(tree.right, depth + 1, d)
        
    return d
        
if __name__ == "__main__":
    # TODO: Uncomment these when you have implemented all the functions
    # import doctest
    # doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
