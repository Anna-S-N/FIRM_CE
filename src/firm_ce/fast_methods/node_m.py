import numpy as np

from firm_ce.common.constants import FASTMATH, TOLERANCE
from firm_ce.common.exceptions import (
    raise_getting_unloaded_data_error,
    raise_static_modification_error,
)
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, boolean, float64, int64, unicode_type
from firm_ce.fast_methods import generator_m
from firm_ce.system.components import Generator_InstanceType, Storage_InstanceType
from firm_ce.system.topology import Node, Node_InstanceType


@njit(fastmath=FASTMATH)
def create_dynamic_copy(node_instance: Node_InstanceType) -> Node_InstanceType:
    """
    A 'static' instance of the Node jitclass (Node.static_instance=True) is copied
    and marked as a 'dynamic' instance (Node.static_instance=False).

    Static instances are created during Model initialisation and supplied as arguments
    to the differential evolution. These arguments are references to the original jitclass instances (not copies).
    Candidate solutions within the differential evolution are tested in embarrasingly parrallel,
    making it unsafe for multiple workers to similtaneously modify the same memory referenced
    across each process.

    Instead, each worker must create a deep copy of the referenced instance that is safe to modify
    within that worker process. Not all attributes within a dynamic instance are safe to modify.
    Only attributes that are required to be modified when testing the candidate solution are
    copied in order to save memory. If an attribute is unsafe to modify after copying, it will
    be marked with a comment that says "This remains static" in the create_dynamic_copy fast_method for
    that jitclass.

    Parameters:
    -------
    node_instance (Node_InstanceType): A static instance of the Node jitclass.

    Returns:
    -------
    Node_InstanceType: A dynamic instance of the Node jitclass.
    """
    node_copy = Node(False, node_instance.id, node_instance.order, node_instance.name)
    node_copy.data_status = node_instance.data_status
    node_copy.data = node_instance.data  # This remains static
    node_copy.residual_load = node_instance.residual_load.copy()
    return node_copy


@njit(fastmath=FASTMATH)
def load_data(node_instance: Node_InstanceType, trace: float64[:]) -> None:
    """
    Load the electricity demand trace and initialise the residual load for a Node instance.

    Parameters:
    -------
    node_instance (Node_InstanceType): An instance of the Node jitclass.
    trace (float64[:]): Array containing the time-series electricity demand trace for the Node. Each element
        provides the demand [MW] for a time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Node instance: data_status, data, residual_load.
    """
    node_instance.data_status = "loaded"
    node_instance.data = trace
    node_instance.residual_load = trace.copy()
    return None


@njit(fastmath=FASTMATH)
def unload_data(node_instance: Node_InstanceType) -> None:
    """
    Unload the electricity demand trace and the residual load from the Node instance. This is done
    after solving a Scenario to reduce memory usage.

    Parameters:
    -------
    node_instance (Node_InstanceType): An instance of the Node jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Node instance: data_status, data, residual_load.
    """
    node_instance.data_status = "unloaded"
    node_instance.data = np.empty((0,), dtype=np.float64)
    node_instance.residual_load = np.empty((0,), dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def get_data(node_instance: Node_InstanceType, data_type: unicode_type) -> float64[:]:
    """
    Gets the specified data_type from the Node instance.

    Parameters:
    -------
    node_instance (Node_InstanceType): An instance of the Node jitclass.
    data_type (unicode_type): String associated with the data array.

    Returns:
    -------
    float64[:]: The data array associated with data_type.

    Raises:
    -------
    RuntimeError: Raised if data_status is "unloaded" or if data_type does not correspond
        to any data arrays for the Node jitclass.
    """
    if node_instance.data_status == "unloaded":
        raise_getting_unloaded_data_error()

    if data_type == "trace":
        return node_instance.data
    elif data_type == "residual_load":
        return node_instance.residual_load
    else:
        raise RuntimeError("Invalid data_type argument for Node.get_data(data_type).")


@njit(fastmath=FASTMATH)
def allocate_memory(node_instance: Node_InstanceType, intervals_count: int64) -> None:
    """
    Memory associated with endogenous time-series data for a Node is only allocated after a dynamic copy of
    the Node instance is created. This is to minimise memory usage of the static instances.

    Parameters:
    -------
    node_instance (Node_InstanceType): A dynamic instance of the Node jitclass.
    intervals_count (int64): Total number of time intervals over the modelling horizon.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Node instance: imports_exports, deficits, spillage, flexible_power, storage_power,
        flexible_energy,  storage_energy.

    Raises:
    -------
    RuntimeError: Raised if static_instance is True. Only dynamic instances can be modified by this pseudo-method.
    """
    if node_instance.static_instance:
        raise_static_modification_error()
    node_instance.imports_exports = np.zeros(intervals_count, dtype=np.float64)
    node_instance.deficits = np.zeros(intervals_count, dtype=np.float64)
    node_instance.spillage = np.zeros(intervals_count, dtype=np.float64)

    node_instance.flexible_power = np.zeros(intervals_count, dtype=np.float64)
    node_instance.storage_power = np.zeros(intervals_count, dtype=np.float64)
    node_instance.flexible_energy = np.zeros(intervals_count, dtype=np.float64)
    node_instance.storage_energy = np.zeros(intervals_count, dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def initialise_netload_t(node_instance: Node_InstanceType, interval: int64) -> None:
    """
    Initialises the netload for a Node to be the residual load. The residual load is equal to the
    operational demand minus generation from solar, wind, and baseload generation in a given time interval.
    The netload is a temporary value that also accounts for imports/exports (as well as storage and flexible
    dispatch when precharging).

    Parameters:
    -------
    node_instance (Node_InstanceType): An instance of the Node jitclass.
    interval (int64): Index for the time interval during unit committment.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Node instance: netload_t.
    """
    node_instance.netload_t = get_data(node_instance, "residual_load")[interval]
    return None


@njit(fastmath=FASTMATH)
def update_netload_t(node_instance: Node_InstanceType, interval: int64, precharging_flag: boolean) -> None:
    """
    Updates the netload for a Node in a given time interval based upon the current imports/exports to that Node.
    When precharging, the storage and flexible dispatch to that node is also accounted for in the netload
    temporary valuable.

    Parameters:
    -------
    node_instance (Node_InstanceType): An instance of the Node jitclass.
    interval (int64): Index for the time interval during unit committment.
    precharging_flag (boolean): True if balancing in either a deficit block or precharging period. Otherwise, False.
        When the value is True, the netload calculation also considers current storage and flexible dispatch at that
        node.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Node instance: netload_t.
    """
    # Note: exports are negative, so they add to load
    node_instance.netload_t = (
        get_data(node_instance, "residual_load")[interval] - node_instance.imports_exports[interval]
    )

    if precharging_flag:
        node_instance.netload_t -= node_instance.storage_power[interval] + node_instance.flexible_power[interval]
    return None


@njit(fastmath=FASTMATH)
def fill_required(node_instance: Node_InstanceType) -> boolean:
    """
    Checks whether the Node has an fill energy that it is attempting to balance through transmission.

    Parameters:
    -------
    node_instance (Node_InstanceType): An instance of the Node jitclass.

    Returns:
    -------
    boolean: True if there is fill energy that the Node is attempting to balance, otherwise False.
    """
    return node_instance.fill > TOLERANCE


@njit(fastmath=FASTMATH)
def surplus_available(node_instance: Node_InstanceType) -> boolean:
    """
    Checks whether the Node has any surplus energy available for transmission.

    Parameters:
    -------
    node_instance (Node_InstanceType): An instance of the Node jitclass.

    Returns:
    -------
    boolean: True if there is surplus energy at the Node available for transmission, otherwise False.
    """
    return node_instance.surplus > TOLERANCE


@njit(fastmath=FASTMATH)
def assign_storage_merit_order(
    node_instance: Node_InstanceType, storages_typed_dict: DictType(int64, Storage_InstanceType)
) -> None:
    """
    Identifies Storage instances located at the Node and sorts them from shortest to longest storage duration.
    The Storage.order values for the merit order are stored in an array for the Node.

    The pseudo-method starts by iterating through all Storage instances in the scenario and adding the Storage.order
    and Storage.duration values to temporary arrays. If there are no Storage instances at the Node, the function
    returns None early. Otherwise, the temporary arrays are clipped to have a length equal to the number of Storage
    instances at that Node. The indices that would sort the temporary storage duration array are calculated, and then
    the temporary storage order array is sorted using those indices.

    Parameters:
    -------
    node_instance (Node_InstanceType): An instance of the Node jitclass.
    storages_typed_dict (DictType(int64, Storage_InstanceType)): Typed dictionary of Storage instances within
        the scenario, keyed by Storage.order.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: storage_merit_order.
    """
    storages_count = len(storages_typed_dict)
    temp_orders = np.full(storages_count, -1, dtype=np.int64)
    temp_durations = np.full(storages_count, -1, dtype=np.float64)

    idx = 0
    for storage_order, storage in storages_typed_dict.items():
        if storage.node.order == node_instance.order:
            temp_orders[idx] = storage_order
            temp_durations[idx] = storage.duration
            idx += 1

    if idx == 0:
        return None

    temp_orders = temp_orders[:idx]
    temp_durations = temp_durations[:idx]

    sort_order = np.argsort(temp_durations)
    node_instance.storage_merit_order = temp_orders[sort_order]
    return None


@njit(fastmath=FASTMATH)
def assign_flexible_merit_order(
    node_instance: Node_InstanceType, generators_typed_dict: DictType(int64, Generator_InstanceType)
) -> None:
    """
    Identifies flexible Generator instances located at the Node and sorts them from cheapest to most expensive marginal
    variable costs. The flexible Generator.order values for the merit order are stored in an array for the Node.

    The pseudo-method starts by iterating through all flexible Generator instances in the scenario and adding the Generator.order
    and marginal variable cost values to temporary arrays. The marginal variable cost is assumed to be equal to the cost of 1 unit
    generating at 1 MW for 1 hour. If there are no flexible Generator instances at the Node, the function
    returns None early. Otherwise, the temporary arrays are clipped to have a length equal to the number of flexible Generator
    instances at that Node. The indices that would sort the temporary marginal variable cost array are calculated, and then
    the temporary flexible generator order array is sorted using those indices.

    Parameters:
    -------
    node_instance (Node_InstanceType): An instance of the Node jitclass.
    generators_typed_dict (DictType(int64, Generators_InstanceType)): Typed dictionary of Generator
        instances within the scenario, keyed by Generator.order.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: flexible_merit_order.
    """
    generators_count = len(generators_typed_dict)
    temp_orders = np.full(generators_count, -1, dtype=np.int64)
    temp_marginal_costs = np.full(generators_count, -1, dtype=np.float64)

    idx = 0
    for generator_order, generator in generators_typed_dict.items():
        if not generator_m.check_unit_type(generator, "flexible"):
            continue

        if generator.node.order == node_instance.order:
            temp_orders[idx] = generator_order
            temp_marginal_costs[idx] = (
                generator.cost.vom
                + generator.cost.fuel_cost_mwh
                + generator.cost.fuel_cost_h * 1000 * generator.unit_size
            )
            idx += 1

    if idx == 0:
        return

    temp_orders = temp_orders[:idx]
    temp_marginal_costs = temp_marginal_costs[:idx]

    sort_order = np.argsort(temp_marginal_costs)
    node_instance.flexible_merit_order = temp_orders[sort_order]
    return None


@njit(fastmath=FASTMATH)
def check_remaining_netload(node_instance: Node_InstanceType, interval: int64, check_case: unicode_type) -> boolean:
    """
    Checks whether there is any remaining unbalanced 'deficit', 'spillage', or 'both' at the Node
    by evaluating its netloads. If the Node has unbalanced netload for the check case, function returns
    a value of True.

    Parameters:
    -------
    node_instance (Node_InstanceType): A dynamic instance of the Node jitclass.
    interval (int64): Index for the time interval.
    check_case (unicode_type): Either 'deficit' (netload greater than 0), 'spillage' (netload less than 0),
        or 'both' (netload equals 0).

    Returns:
    -------
    boolean: True if the Node has unbalanced netload according to the check case, otherwise False.
    """
    if check_case == "deficit":
        return (
            node_instance.netload_t - node_instance.storage_power[interval] - node_instance.flexible_power[interval]
            > TOLERANCE
        )
    elif check_case == "spillage":
        return (
            node_instance.netload_t - node_instance.storage_power[interval] - node_instance.flexible_power[interval]
            < -TOLERANCE
        )
    elif check_case == "both":
        return (
            abs(
                node_instance.netload_t - node_instance.storage_power[interval] - node_instance.flexible_power[interval]
            )
            > TOLERANCE
        )
    return False


@njit(fastmath=FASTMATH)
def set_imports_exports_temp(node_instance: Node_InstanceType, interval: int64) -> None:
    """
    During the precharging period, imports/exports at the nodes are adjusted to allow for additional transmission
    for precharging of Storage systems. The temporary imports/exports value is used to store the current value of
    imports/exports prior to adjusting the transmission. This allows the change in imports/exports following a
    transmission action to be calculated.

    Parameters:
    -------
    node_instance (Node_InstanceType): A dynamic instance of the Node jitclass.
    interval (int64): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: imports_exports_temp.
    """
    node_instance.imports_exports_temp = node_instance.imports_exports[interval]
    return None


@njit(fastmath=FASTMATH)
def reset_dispatch_max_t(node_instance: Node_InstanceType) -> None:
    """
    Resets the temporary storage discharging/charging maximum powers and flexible maximum powers for the node
    to zero before balancing a new time interval. Note that these nodal values are cumulative sums for Storage systems
    and flexible Generators along the merit order for this Node (with the cumulative sums stored in an array).
    If there are no Storage systems (or flexible Generators) at the Node, then the arrays have a length of 1.

    Parameters:
    -------
    node_instance (Node_InstanceType): A dynamic instance of the Node jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: discharge_max_t, charge_max_t, flexible_max_t.
    """
    if len(node_instance.storage_merit_order) > 0:
        node_instance.discharge_max_t = np.zeros(len(node_instance.storage_merit_order), dtype=np.float64)
        node_instance.charge_max_t = np.zeros(len(node_instance.storage_merit_order), dtype=np.float64)
    else:
        node_instance.discharge_max_t = np.zeros(1, dtype=np.float64)
        node_instance.charge_max_t = np.zeros(1, dtype=np.float64)

    if len(node_instance.flexible_merit_order) > 0:
        node_instance.flexible_max_t = np.zeros(len(node_instance.flexible_merit_order), dtype=np.float64)
    else:
        node_instance.flexible_max_t = np.zeros(1, dtype=np.float64)
    return None
