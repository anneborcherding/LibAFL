def bytes_as_bits_list(input_bytes):
    """returns a list of floats representing the bits of the given bytearray"""
    bytes_as_bits = ''.join(format(byte, '08b') for byte in input_bytes)
    # make sure there are no decimal places (kinda ugly maybe)
    return list([float(int(float(a))) for a in bytes_as_bits])


def scale_packet_values(input_bytes):
    """returns a list of floats representing the bytes of the given bytearray scaled to [0,1].
        Idea from Chiu et al. (2020)"""
    return [byte / 255 for byte in input_bytes]


def scale_packet_length(input_bytes, length=1500):
    """returns the bytes trimmed or padded to the given length (in bytes).
        Idea from Chiu et al. (2020)"""
    return input_bytes.ljust(length, b'\0') if len(input_bytes) <= length else input_bytes[0:length]