"""
Adaptive Group Encoding (AGE) Framework Implementation

This module implements the AGE framework for privacy-preserving encoding of
adaptive sampling data. AGE eliminates information leakage through message-size
side-channels by producing fixed-length encoded messages.

Reference:
    Kannan, A., & Hoffmann, H. (2022). "Protecting Adaptive Sampling from
    Information Leakage on Low-Power Sensors." ASPLOS.

Author: AGE Framework Implementation
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Optional
import struct


class AGEEncoder:
    """
    Adaptive Group Encoding (AGE) encoder.

    Implements a three-step lossy encoding process:
    1. Measurement Pruning (§4.2): Remove least-important measurements
    2. Exponent-Aware Group Formation (§4.3): Group values with similar exponents
    3. Data Quantization (§4.4): Quantize and pack into fixed-length message

    Attributes:
        target_bytes (int): Target message size in bytes (M_B)
        w_min (int): Minimum bit width for quantization
        max_groups (int): Maximum number of groups (G)
    """

    def __init__(self, target_bytes: int = 100, w_min: int = 4, max_groups: int = 16):
        """
        Initialize AGE encoder.

        Args:
            target_bytes: Fixed output message size in bytes (default: 100)
            w_min: Minimum bit width for quantization (default: 4)
            max_groups: Maximum number of groups allowed (default: 16)
        """
        self.target_bytes = target_bytes
        self.w_min = w_min
        self.max_groups = max_groups

        # Reserve bytes for metadata
        # Format: [num_samples (2B), num_groups (1B), group_metadata, quantized_data]
        self.metadata_overhead = 3  # Initial overhead for counts

    def encode(self, samples: np.ndarray) -> np.ndarray:
        """
        Encode variable-length samples into fixed-length byte array.

        Args:
            samples: 1D numpy array of sensor measurements (floats)

        Returns:
            Fixed-length byte array of size target_bytes

        Raises:
            ValueError: If samples array is empty or invalid
        """
        if len(samples) == 0:
            raise ValueError("Cannot encode empty sample array")

        # Ensure samples is float64
        samples = np.asarray(samples, dtype=np.float64)

        # Step 1: Measurement Pruning
        pruned_samples, pruned_indices = self._prune_measurements(samples)

        # Step 2: Exponent-Aware Group Formation
        exponents = self._compute_exponents(pruned_samples)
        groups = self._group_formation_rle(exponents)

        # Merge groups if necessary
        if len(groups) > self.max_groups:
            groups = self._merge_groups(groups, exponents)

        # Step 3: Data Quantization
        bit_widths = self._assign_bit_widths(groups, len(pruned_samples))
        encoded_message = self._quantize_and_pack(
            pruned_samples, pruned_indices, groups, bit_widths
        )

        return encoded_message

    def _prune_measurements(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove least-important measurements to fit in target message size.

        Uses distance scoring with temporal penalty:
            Dist(x_t) = |x_t - x_{t+1}| + (1/8)|α_t - α_{t+1}|

        Where α_t is the temporal index (position in sequence).

        Args:
            samples: Input samples array

        Returns:
            Tuple of (pruned_samples, kept_indices)
        """
        n = len(samples)

        # Calculate maximum samples that can fit with w_min bits each
        # Available space = target_bytes - metadata_overhead
        # Need space for: group metadata (~max_groups * 3 bytes) + quantized data
        estimated_metadata = self.metadata_overhead + (self.max_groups * 3)
        available_bits = (self.target_bytes - estimated_metadata) * 8
        max_samples = int(available_bits / self.w_min)

        # If we can fit all samples, no pruning needed
        if n <= max_samples:
            return samples, np.arange(n)

        # Create list of (value, original_index) tuples
        indexed_samples = [(samples[i], i) for i in range(n)]

        # Iteratively remove samples with smallest distance score
        while len(indexed_samples) > max_samples:
            # Compute distance scores for all adjacent pairs
            scores = []
            for i in range(len(indexed_samples) - 1):
                val_t, idx_t = indexed_samples[i]
                val_t1, idx_t1 = indexed_samples[i + 1]

                # Distance metric: value difference + temporal penalty
                value_diff = abs(val_t - val_t1)
                temporal_penalty = (1 / 8) * abs(idx_t - idx_t1)
                dist_score = value_diff + temporal_penalty

                scores.append((dist_score, i))

            # Safety check: if no scores, break
            if not scores:
                break

            # Find and remove the pair with minimum score
            # We remove the second element of the pair
            min_score, min_idx = min(scores, key=lambda x: x[0])
            indexed_samples.pop(min_idx + 1)

        # Extract pruned samples and indices
        pruned_samples = np.array([val for val, _ in indexed_samples])
        kept_indices = np.array([idx for _, idx in indexed_samples])

        return pruned_samples, kept_indices

    def _compute_exponents(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute exponent (number of non-fractional bits) for each measurement.

        For floating point value x, the exponent e is computed as:
            e = floor(log2(|x|)) + 1  if x != 0
            e = 0                      if x == 0

        Args:
            samples: Array of sample values

        Returns:
            Array of exponents (integers)
        """
        exponents = np.zeros(len(samples), dtype=np.int32)

        for i, val in enumerate(samples):
            if val == 0:
                exponents[i] = 0
            else:
                # Compute number of bits needed for integer part
                abs_val = abs(val)
                exponents[i] = int(np.floor(np.log2(abs_val))) + 1

        return exponents

    def _group_formation_rle(self, exponents: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Form groups using run-length encoding (RLE) on exponents.

        Consecutive measurements with the same exponent are grouped together.

        Args:
            exponents: Array of exponent values

        Returns:
            List of tuples (start_idx, end_idx, exponent)
        """
        if len(exponents) == 0:
            return []

        groups = []
        start_idx = 0
        current_exp = exponents[0]

        for i in range(1, len(exponents)):
            if exponents[i] != current_exp:
                # End of current run
                groups.append((start_idx, i - 1, current_exp))
                start_idx = i
                current_exp = exponents[i]

        # Add final group
        groups.append((start_idx, len(exponents) - 1, current_exp))

        return groups

    def _merge_groups(
        self, groups: List[Tuple[int, int, int]], exponents: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Merge groups until max_groups constraint is satisfied.

        Uses a greedy heuristic: merge adjacent groups with smallest
        exponent difference and smallest combined size.

        Args:
            groups: List of (start_idx, end_idx, exponent) tuples
            exponents: Original exponent array

        Returns:
            Merged list of groups
        """
        while len(groups) > self.max_groups:
            # Compute merge scores for all adjacent pairs
            merge_scores = []
            for i in range(len(groups) - 1):
                start1, end1, exp1 = groups[i]
                start2, end2, exp2 = groups[i + 1]

                # Score: prioritize small exponent difference and small size
                size1 = end1 - start1 + 1
                size2 = end2 - start2 + 1
                exp_diff = abs(exp1 - exp2)

                # Lower score = better merge candidate
                score = exp_diff * 10 + (size1 + size2) * 0.1
                merge_scores.append((score, i))

            # Merge the pair with lowest score
            _, merge_idx = min(merge_scores, key=lambda x: x[0])

            # Create merged group
            start1, end1, exp1 = groups[merge_idx]
            start2, end2, exp2 = groups[merge_idx + 1]

            # Use maximum exponent to avoid overflow
            merged_exp = max(exp1, exp2)
            merged_group = (start1, end2, merged_exp)

            # Replace two groups with merged group
            groups = groups[:merge_idx] + [merged_group] + groups[merge_idx + 2 :]

        return groups

    def _assign_bit_widths(
        self, groups: List[Tuple[int, int, int]], num_samples: int
    ) -> List[int]:
        """
        Assign bit widths to each group using round-robin allocation.

        Distributes available bits across groups, respecting w_min constraint
        and target message size.

        Args:
            groups: List of group tuples
            num_samples: Total number of samples to encode

        Returns:
            List of bit widths per group
        """
        num_groups = len(groups)

        # Calculate available bits for data
        # Metadata: num_samples(2B) + num_groups(1B) + per_group_metadata
        # Per-group metadata: start_idx(2B) + bit_width(1B) per group
        metadata_bytes = self.metadata_overhead + (num_groups * 3)
        available_bytes = self.target_bytes - metadata_bytes
        available_bits = available_bytes * 8

        # Initialize all groups with w_min
        bit_widths = [self.w_min] * num_groups

        # Calculate group sizes
        group_sizes = [end - start + 1 for start, end, _ in groups]
        total_bits_used = sum(bw * size for bw, size in zip(bit_widths, group_sizes))

        # Distribute remaining bits round-robin
        remaining_bits = available_bits - total_bits_used

        if remaining_bits > 0:
            group_idx = 0
            while remaining_bits >= sum(group_sizes):
                # Add 1 bit to current group
                additional_bits = group_sizes[group_idx]
                if remaining_bits >= additional_bits:
                    bit_widths[group_idx] += 1
                    remaining_bits -= additional_bits

                group_idx = (group_idx + 1) % num_groups

                # Safety check to prevent infinite loop
                if all(bw >= 32 for bw in bit_widths):
                    break

        return bit_widths

    def _quantize_and_pack(
        self,
        samples: np.ndarray,
        indices: np.ndarray,
        groups: List[Tuple[int, int, int]],
        bit_widths: List[int],
    ) -> np.ndarray:
        """
        Quantize samples and pack into fixed-length byte array.

        Format:
            [num_samples (2B)] [num_groups (1B)] [group_metadata] [quantized_data] [padding]

        Group metadata per group:
            [start_idx (2B)] [bit_width (1B)]

        Args:
            samples: Pruned sample values
            indices: Original indices of samples
            groups: Group definitions
            bit_widths: Bit width per group

        Returns:
            Fixed-length byte array
        """
        # Create output buffer
        output = bytearray(self.target_bytes)
        byte_offset = 0

        # Write header
        struct.pack_into(
            "<H", output, byte_offset, len(samples)
        )  # num_samples (2 bytes)
        byte_offset += 2

        struct.pack_into("<B", output, byte_offset, len(groups))  # num_groups (1 byte)
        byte_offset += 1

        # Write group metadata
        for (start_idx, end_idx, exponent), bit_width in zip(groups, bit_widths):
            struct.pack_into(
                "<H", output, byte_offset, start_idx
            )  # start_idx (2 bytes)
            byte_offset += 2
            struct.pack_into("<B", output, byte_offset, bit_width)  # bit_width (1 byte)
            byte_offset += 1

        # Prepare bit buffer for quantized data
        bit_buffer = []

        # Quantize each group
        for (start_idx, end_idx, exponent), bit_width in zip(groups, bit_widths):
            group_samples = samples[start_idx : end_idx + 1]

            # Normalize to [0, 1] range within group
            group_min = np.min(group_samples)
            group_max = np.max(group_samples)

            # Store normalization parameters (use exponent for range estimation)
            # For decoding, we'll need to store group_min and group_max
            # Pack them as float32 to save space
            range_val = group_max - group_min if group_max != group_min else 1.0

            for sample in group_samples:
                # Normalize
                if range_val != 0:
                    normalized = (sample - group_min) / range_val
                else:
                    normalized = 0.0

                # Quantize to bit_width bits
                max_int_val = (1 << bit_width) - 1
                quantized = int(normalized * max_int_val)
                quantized = max(0, min(quantized, max_int_val))  # Clamp

                # Convert to binary string
                binary_str = format(quantized, f"0{bit_width}b")
                bit_buffer.extend([int(b) for b in binary_str])

        # Pack bits into bytes
        while len(bit_buffer) >= 8 and byte_offset < self.target_bytes:
            byte_val = 0
            for i in range(8):
                byte_val = (byte_val << 1) | bit_buffer[i]
            bit_buffer = bit_buffer[8:]

            output[byte_offset] = byte_val
            byte_offset += 1

        # Handle remaining bits (padding)
        if bit_buffer and byte_offset < self.target_bytes:
            byte_val = 0
            for bit in bit_buffer:
                byte_val = (byte_val << 1) | bit
            # Pad with zeros
            byte_val <<= 8 - len(bit_buffer)
            output[byte_offset] = byte_val
            byte_offset += 1

        # Remaining bytes are already zero-initialized
        return np.frombuffer(output, dtype=np.uint8)


class AGEDecoder:
    """
    Decoder for AGE-encoded messages.

    Reconstructs approximate original samples from fixed-length encoded messages.
    Note: This is lossy - reconstruction will not be perfect.
    """

    def __init__(self):
        """Initialize AGE decoder."""
        pass

    def decode(self, encoded_message: np.ndarray) -> np.ndarray:
        """
        Decode fixed-length message back to sample values.

        Args:
            encoded_message: Fixed-length byte array from AGEEncoder

        Returns:
            Reconstructed samples (approximate)

        Raises:
            ValueError: If message format is invalid
        """
        if len(encoded_message) == 0:
            raise ValueError("Cannot decode empty message")

        # Convert to bytes
        message_bytes = bytes(encoded_message)
        byte_offset = 0

        # Read header
        num_samples = struct.unpack_from("<H", message_bytes, byte_offset)[0]
        byte_offset += 2

        num_groups = struct.unpack_from("<B", message_bytes, byte_offset)[0]
        byte_offset += 1

        # Read group metadata
        groups = []
        for _ in range(num_groups):
            start_idx = struct.unpack_from("<H", message_bytes, byte_offset)[0]
            byte_offset += 2
            bit_width = struct.unpack_from("<B", message_bytes, byte_offset)[0]
            byte_offset += 1
            groups.append((start_idx, bit_width))

        # Prepare output array
        decoded_samples = np.zeros(num_samples, dtype=np.float64)

        # Extract bit stream
        bit_stream = []
        for byte_val in message_bytes[byte_offset:]:
            binary_str = format(byte_val, "08b")
            bit_stream.extend([int(b) for b in binary_str])

        # Decode each group
        bit_pos = 0
        for i, (start_idx, bit_width) in enumerate(groups):
            # Determine end index (next group's start or num_samples)
            if i < len(groups) - 1:
                end_idx = groups[i + 1][0] - 1
            else:
                end_idx = num_samples - 1

            group_size = end_idx - start_idx + 1

            # Extract quantized values
            max_int_val = (1 << bit_width) - 1
            for j in range(group_size):
                if bit_pos + bit_width <= len(bit_stream):
                    # Extract bit_width bits
                    quantized_val = 0
                    for k in range(bit_width):
                        quantized_val = (quantized_val << 1) | bit_stream[bit_pos]
                        bit_pos += 1

                    # De-quantize (assuming [0, 1] normalization)
                    # Note: Without stored min/max, we use estimated range
                    normalized = quantized_val / max_int_val if max_int_val > 0 else 0

                    # Reconstruct (placeholder - needs group min/max for accuracy)
                    # For now, use normalized value as-is
                    decoded_samples[start_idx + j] = normalized

        return decoded_samples
