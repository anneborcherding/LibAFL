import typing
from pathlib import Path
import traceback

from scapy.sendrecv import sniff
from scapy.layers.inet import TCP, IP
from scapy.all import raw


def load_data(path: Path, filter_expr: str = "") -> typing.Any:
    """
    :param path: Path to the pcap file that should be loaded.
    :param filter_expr: BPF filter expression that will be used to filter the loaded network traffic.
    :return: list of packets that satisfy the filter condition.
    """
    try:
        ps = sniff(offline=str(path), filter=filter_expr)
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return ps

def load_and_split_data_syn(path : Path, filter_expr : str =""):
    """
    Loads the data at the given path and split it based on the seen SYN packets.
    :param path: Path to the pcap.
    :param filter_expr: BPF filter expression that will be used to filter the loaded network traffic.
    :return: List of packets.
    """
    packets_filtered = load_data(path, filter_expr)
    packets_filtered_split = {}
    current_sequence = []
    sequence_no = 0

    for pkt in packets_filtered:
        if IP in pkt and TCP in pkt and str(pkt[TCP].flags) == 'S':
            # new sequence
            if len(current_sequence) > 0:
                packets_filtered_split[sequence_no] = current_sequence
                current_sequence = []
                sequence_no = sequence_no + 1
        current_sequence.append(raw(pkt))
    packets_filtered_split[sequence_no] = current_sequence
    return packets_filtered_split

def load_and_split_data_ports(path : Path, ip_addr_sender : str, ip_addr_target : str, filter_expr : str =""):
    """
    Loads the data at the given path and split it based on the seen ports. Beware: If the network capture is long enough, a port might have been reused and as a result, several communication relationships are put together.
    :param path: Path to the pcap.
    :param filter_expr: BPF filter expression that will be used to filter the loaded network traffic.
    :param ip_addr_target: IP address of the target, used to filter the traffic.
    :param ip_addr_sender: IP address of the sender, used to filter the traffic.
    :return: List of packets.
    """
    packets_filtered = load_data(path, filter_expr)
    seen_ports = []
    packets_filtered_split = {}

    for pkt in packets_filtered:
        if IP in pkt and TCP in pkt:
            # TODO think more deeply about what happens if sender ip == target ip
            if pkt[IP].src == ip_addr_sender:
                port = pkt[TCP].sport
            elif pkt[IP].src == ip_addr_target:
                port = pkt[TCP].dport
            else:
                #should not happen
                continue
            if port in seen_ports:
                packets_filtered_split[str(port)].append(raw(pkt))
            else:
                seen_ports.append(port)
                packets_filtered_split[str(port)] = [raw(pkt)]
    return packets_filtered_split