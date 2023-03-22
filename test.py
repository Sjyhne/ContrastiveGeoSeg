import torch

def extract_edges_batch(masks, class_id):
    assert masks.dim() == 3, "masks should have 3 dimensions: N x H x W"

    if torch.cuda.is_available():
        masks = masks.cuda()

    # Create a binary mask for the specified class
    binary_masks = (masks == class_id).float()

    # Calculate the difference between neighboring pixels
    horizontal_diff = binary_masks[:, :, 1:] - binary_masks[:, :, :-1]
    vertical_diff = binary_masks[:, 1:, :] - binary_masks[:, :-1, :]

    # Create the edge masks
    edge_masks = torch.zeros_like(binary_masks)
    edge_masks[:, :, 1:] += horizontal_diff.abs()
    edge_masks[:, :, :-1] += horizontal_diff.abs()
    edge_masks[:, 1:, :] += vertical_diff.abs()
    edge_masks[:, :-1, :] += vertical_diff.abs()

    # Convert the edge masks to boolean tensors
    edge_masks = (edge_masks > 0)
    
    #edge_masks = edge_masks.to(torch.int32)

    return edge_masks

masks = torch.tensor([
    [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2]
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
])

# Extract the edges for class 1
class_id = 0
edge_masks = extract_edges_batch(masks, class_id)

# Print the edge masks
print(edge_masks)