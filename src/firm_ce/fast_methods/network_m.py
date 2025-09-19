from firm_ce.common.constants import FASTMATH, TOLERANCE
from firm_ce.common.exceptions import raise_static_modification_error
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import TypedDict, TypedList, boolean, float64, int64, unicode_type
from firm_ce.fast_methods import line_m, node_m, route_m
from firm_ce.system.topology import (
    Line_InstanceType,
    Network,
    Network_InstanceType,
    Node_InstanceType,
    Route_InstanceType,
    routes_key_type,
    routes_list_type,
)


@njit(fastmath=FASTMATH)
def create_dynamic_copy(network_instance: Network_InstanceType) -> Network_InstanceType:
    """
    A 'static' instance of the Network jitclass (Network.static_instance=True) is copied
    and marked as a 'dynamic' instance (Network.static_instance=False).

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
    network_instance (Network_InstanceType): A static instance of the Fleet jitclass.

    Returns:
    -------
    Network_InstanceType: A dynamic instance of the Network jitclass.
    """
    nodes_copy = TypedDict.empty(key_type=int64, value_type=Node_InstanceType)
    major_lines_copy = TypedDict.empty(key_type=int64, value_type=Line_InstanceType)
    minor_lines_copy = TypedDict.empty(key_type=int64, value_type=Line_InstanceType)
    routes_copy = TypedDict.empty(key_type=routes_key_type, value_type=routes_list_type)

    for order, node in network_instance.nodes.items():
        nodes_copy[order] = node_m.create_dynamic_copy(node)

    for order, line in network_instance.major_lines.items():
        major_lines_copy[order] = line_m.create_dynamic_copy(line, nodes_copy, "major")

    for order, line in network_instance.minor_lines.items():
        minor_lines_copy[order] = line_m.create_dynamic_copy(line, nodes_copy, "minor")

    for tuple_key, routes_list in network_instance.routes.items():
        routes_copy[tuple_key] = TypedList.empty_list(Route_InstanceType)
        for route in routes_list:
            routes_copy[tuple_key].append(route_m.create_dynamic_copy(route, nodes_copy, major_lines_copy))

    network_copy = Network(
        False,
        nodes_copy,
        major_lines_copy,
        minor_lines_copy,
        routes_copy,
        network_instance.networksteps_max,
    )
    network_copy.major_line_count = network_instance.major_line_count
    return network_copy


@njit(fastmath=FASTMATH)
def build_capacity(network_instance: Network_InstanceType, decision_x: float64[:]) -> None:
    """
    The candidate solution defines new build capacity for each Generator, Storage, and Line (major_lines) object. This
    function modifies each major_line Line object in the Network to build new capacity.

    Note that the capacity for minor_lines is built at the time that Generator and Storage assets build capacity.

    Parameters:
    -------
    network_instance (Network_InstanceType): A dynamic instance of the Network jitclass.
    decision_x (float64[:]): A 1-dimensional array containing the candidate solution for the differential
        evolution. The candidate solution defines new build capacity for each decision variable (either power
        or energy capacity).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Line instance in Network.major_lines: new_build, capacity.

    Raises:
    -------
    RuntimeError: Raised if static_instance is True. Only dynamic instances can be modified by this pseudo-method.
    """
    if network_instance.static_instance:
        raise_static_modification_error()
    for line in network_instance.major_lines.values():
        line_m.build_capacity(line, decision_x[line.candidate_x_idx])
    return None


@njit(fastmath=FASTMATH)
def unload_data(network_instance: Network_InstanceType) -> None:
    """
    Load the operational demand trace for each Node instance in Network.nodes and initialise the
    residual_load at those Nodes. This is done before solving a Scenario and before building Fleet
    capacities.

    Parameters:
    -------
    network_instance (Network_InstanceType): A dynamic instance of the Network jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node instance in Network.nodes: data_status, data, residual_load.
    """
    for node in network_instance.nodes.values():
        node_m.unload_data(node)
    return None


@njit(fastmath=FASTMATH)
def allocate_memory(network_instance: Network_InstanceType, intervals_count: int64) -> None:
    """
    Memory associated with endogenous time-series data for Node and Line instances is only
    allocated after a dynamic copy of the Network instance is created. This is to minimise memory
    usage of the static instances.

    Parameters:
    -------
    network_instance (Network_InstanceType): A dynamic instance of the Network jitclass.
    intervals_count (int64): Total number of time intervals over the modelling horizon.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node instance in Network.nodes: imports_exports, spillage, deficits, flexible_power,
        storage_power, flexible_energy, storage_energy.
    Attributes modified for each Line instance in Network.major_lines: flows.

    Raises:
    -------
    RuntimeError: Raised if static_instance is True. Only dynamic instances can be modified by this pseudo-method.
    """
    if network_instance.static_instance:
        raise_static_modification_error()
    for node in network_instance.nodes.values():
        node_m.allocate_memory(node, intervals_count)
    for line in network_instance.major_lines.values():
        line_m.allocate_memory(line, intervals_count)
    return None


@njit(fastmath=FASTMATH)
def check_remaining_netloads(
    network_instance: Network_InstanceType, interval: int64, check_case: unicode_type
) -> boolean:
    """
    Checks whether there is any remaining unbalanced 'deficit', 'spillage', or 'both' at each Node
    by evaluating their netloads. If any node has unbalanced netload for the check case, function returns
    a value of True.

    Parameters:
    -------
    network_instance (Network_InstanceType): A dynamic instance of the Network jitclass.
    interval (int64): Index for the time interval.
    check_case (unicode_type): Either 'deficit' (netload greater than 0), 'spillage' (netload less than 0),
        or 'both' (netload equals 0).

    Returns:
    -------
    boolean: True if any Node has unbalanced netload according to the check case, otherwise False.
    """
    for node in network_instance.nodes.values():
        if node_m.check_remaining_netload(node, interval, check_case):
            return True
    return False


@njit(fastmath=FASTMATH)
def calculate_period_unserved_energy(
    network_instance: Network_InstanceType, first_t: int64, last_t: int64, interval_resolutions: float64[:]
) -> float64:
    """
    Calculates the unserved energy at all Node instances between two time periods within the modelling horizon.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    first_t (int64): Index for the first time interval of the specified period.
    last_t (int64): Index for the final time interval of the specified period.
    interval_resolutions (float64[:]): A 1-dimensional array containing the resolution for every time interval
        in the unit committment formulation (hours per time interval). An array is used instead of a single
        scalar value to allow for variable time step simplified balancing methods to be developed in future.

    Returns:
    -------
    float64: Total unserved energy in the specified time period across all Node instances.
    """
    unserved_energy = 0
    for node in network_instance.nodes.values():
        unserved_energy += sum(node.deficits[first_t : last_t + 1] * interval_resolutions[first_t : last_t + 1])
    return unserved_energy


@njit(fastmath=FASTMATH)
def reset_transmission(network_instance: Network_InstanceType, interval: int64) -> None:
    """
    Transmission flows and nodal import/export values are iteratively updated when solving the unit committment
    problem for a given time interval. This function resets all transmission variables to zero for a time interval.
    When a Node instance's import/export value is reset, the netload at must also be recalculated for that Node.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    interval (int64): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Line instance in Network.lines: flows.
    Attributes modified for each Node instance in Network.nodes: imports_exports, netload_t.
    """
    for line in network_instance.major_lines.values():
        line.flows[interval] = 0.0
    for node in network_instance.nodes.values():
        node.imports_exports[interval] = 0.0
        node_m.update_netload_t(node, interval, False)
    return None


@njit(fastmath=FASTMATH)
def reset_flow_updates(network_instance: Network_InstanceType) -> None:
    """
    The flow updates for each Route are a temporary value for a transmission action that
    is used to adjust the flow of all Lines (and import/export of first and last Nodes) along
    the Route. This function resets the temporary flow update value for all Route instances in
    the Network.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Route instance in the route lists (each list corresponds to
        a particular start node and route length) contained in Network.routes: flow_update.
    """
    for route_list in network_instance.routes.values():
        for route in route_list:
            route.flow_update = 0.0
    return None


@njit(fastmath=FASTMATH)
def check_route_surpluses(network_instance: Network_InstanceType, fill_node: Node_InstanceType, leg: int64) -> boolean:
    """
    Checks each Route in a route list (corresponding to a particular start node and route length) and determines if
    the final Node in any Route has surplus energy available. Transmission actions attempt to transmit surpluses to
    balance fills.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    fill_node (Node_InstanceType): The first node in the Route instances in the route list.
    leg (int64): The length of the Route instances in the route list (corresponding to the number of lines in the
        Route).

    Returns:
    -------
    boolean: If any Route in the route list ends in Node containing a surplus, a True value is returned. Otherwise,
        returns False.
    """
    for route in network_instance.routes[fill_node.order, leg]:
        if node_m.surplus_available(route.nodes[-1]):
            return True
    return False


@njit(fastmath=FASTMATH)
def check_network_surplus(network_instance: Network_InstanceType) -> boolean:
    """
    Checks whether any Node in the Network has surplus energy available. Transmission actions attempt to transmit
    surpluses to balance fills.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    boolean: If any Node contains a surplus, a True value is returned. Otherwise, returns False.
    """
    for node in network_instance.nodes.values():
        if node_m.surplus_available(node):
            return True
    return False


@njit(fastmath=FASTMATH)
def check_network_fill(network_instance: Network_InstanceType) -> boolean:
    """
    Checks whether any Node in the Network is attempting to balance fill energy. Transmission actions
    attempt to transmit surpluses to balance fills.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    boolean: If any Node is attempting to balance fill energy, a True value is returned. Otherwise, returns False.
    """
    for node in network_instance.nodes.values():
        if node_m.fill_required(node):
            return True
    return False


@njit(fastmath=FASTMATH)
def calculate_node_flow_updates(
    network_instance: Network_InstanceType, fill_node: Node_InstanceType, leg: int64, interval: int64
) -> None:
    """
    Calculate the maximum possible flow update for each Route. The sum of all flow updates for Routes
    in the route list equal the available imports for the starting node.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    fill_node (Node_InstanceType): The first node in the Route instances in the route list.
    leg (int64): The length of the Route instances in the route list (corresponding to the number of lines in the
        Route).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the fill_node: available_imports is reset to zero at the start of the function, then
        increased in route_m.calculate_flow_update.
    Attributes modified for each Route instance in the route list (the list corresponds to
        a particular start node and route length) contained in Network.routes: flow_update, initial_node (reference to
        same instance as fill_node), nodes, lines.
    Attributes modified for Node referenced by Route.nodes[-1]: temp_surplus.
    Attributes modified for all Line instances referenced in Route.lines: temp_leg_flows.
    """
    fill_node.available_imports = 0.0
    for route in network_instance.routes[fill_node.order, leg]:
        route_m.calculate_flow_update(route, interval)
    return None


@njit(fastmath=FASTMATH)
def scale_flow_updates_to_fill(
    network_instance: Network_InstanceType, fill_node: Node_InstanceType, leg: int64
) -> float64:
    """
    If the available imports for the first node in all Route instances in a route list are greater than
    the fill energy that node is trying to balance, scale all of the Route flow updates down. The sum of
    the scaled Route flow updates equals the fill energy for the first node in the Routes.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    fill_node (Node_InstanceType): The first node in the Route instances in the route list.
    leg (int64): The length of the Route instances in the route list (corresponding to the number of lines in the
        Route).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Route instance in the route list (the list corresponds to
        a particular start node and route length) contained in Network.routes: flow_update.
    """
    if fill_node.available_imports > fill_node.fill:
        scale_factor = fill_node.fill / fill_node.available_imports
        for route in network_instance.routes[fill_node.order, leg]:
            route.flow_update *= scale_factor
    return None


@njit(fastmath=FASTMATH)
def update_transmission_flows(
    network_instance: Network_InstanceType, fill_node: Node_InstanceType, leg: int64, interval: int64
) -> None:
    """
    Once the flow updates for the Route instances corresponding to a particular start node and route length
    have been determined, the transmission action is executed. Flow updates add to the imports at the start
    node during the specified time interval and reduce the fill energy that Node is attempting to balance.
    The Node instances at the end of each Route supply the surplus generation for balancing, so their exports
    are increased and surplus energy reduced by the flow updates. Updates to line flows have equal magnitude
    to the flow updates of the corresponding Route, with the direction defined by the exogenous definition
    of the start and end Node of the Line.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    fill_node (Node_InstanceType): The initial node in the Route instances in the route list.
    leg (int64): The length of the Route instances in the route list (corresponding to the number of lines in the
        Route).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the fill_node: imports_exports (positive is imports, negative is exports), fill.
    Attributes modified for Node referenced by Route.nodes[-1]: imports_exports (positive is imports, negative is exports),
        surplus.
    Attributes modified for all Line instances referenced in Route.lines: flows.
    """
    for route in network_instance.routes[fill_node.order, leg]:
        fill_node.imports_exports[interval] += route.flow_update
        fill_node.fill -= route.flow_update
        route_m.update_exports(route, interval)
    return None


@njit(fastmath=FASTMATH)
def update_netloads(network_instance: Network_InstanceType, interval: int64, precharging_flag: boolean) -> None:
    """
    Recalculate the netload for each Node in the network.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    interval (int64): Index for the time interval.
    precharging_flag (boolean): True if balancing in either a deficit block or precharging period. Otherwise, False.
        When the value is True, the netload calculation also considers current storage and flexible dispatch at that
        node.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: netload_t.
    """
    for node in network_instance.nodes.values():
        node_m.update_netload_t(node, interval, precharging_flag)
    return None


@njit(fastmath=FASTMATH)
def reset_line_temp_flows(network_instance: Network_InstanceType) -> None:
    """
    Reset the temporary value for committed flows for each major Line in the Network to zero.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Line in Network.major_lines: temp_leg_flows.
    """
    for line in network_instance.major_lines.values():
        line.temp_leg_flows = 0.0
    return None


@njit(fastmath=FASTMATH)
def reset_node_temp_surpluses(network_instance: Network_InstanceType) -> None:
    """
    Reset the temporary value for committed surplus energy for each Node in the Network to the total
    surplus for that Node.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: temp_surplus.
    """
    for node in network_instance.nodes.values():
        node.temp_surplus = node.surplus
    return None


@njit(fastmath=FASTMATH)
def fill_with_transmitted_surpluses(network_instance: Network_InstanceType, interval: int64) -> None:
    """
    Overarching function that manages transmission actions. Surplus energy from Nodes is transmitted to
    balance fill energy at other Nodes.

    Transmission is prioritised for nearest neighbours (keyed by the leg variable). A maximum number length
    Route considered for transmission is set by the Network.networksteps_max attribute and can be defined
    in the `scenarios.csv` config file.

    For each fill node in a transmission leg, the maximum available imports (and Route flow updates) is
    calculated. The flow updates are then scaled such that imports from all Routes in the leg sum to equal
    the fill energy for the starting node. The imports/exports for the fill node (first node in each Route
    for that leg) and source nodes (final node in the Route), as well as flows through the transmission
    lines along each Route, are updated.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    interval (int64): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: available_imports, fill, imports_exports.
    Attributes modified for each Route instance in the route lists (the lists corresponds to
        a particular start node and route length) contained in Network.routes: flow_update, initial_node, nodes, lines.
    Attributes modified for Node referenced by Route.nodes[-1]: temp_surplus, surplus, imports_exports.
    Attributes modified for all Line instances referenced in Route.lines: temp_leg_flows, flows.
    """
    reset_flow_updates(network_instance)
    if not (check_network_surplus(network_instance) and check_network_fill(network_instance)):
        return None

    for leg in range(network_instance.networksteps_max):
        for node in network_instance.nodes.values():
            if not node_m.fill_required(node):
                continue
            if len(network_instance.routes[node.order, leg]) == 0:
                continue
            if not check_route_surpluses(network_instance, node, leg):
                continue
            reset_line_temp_flows(network_instance)
            reset_node_temp_surpluses(network_instance)
            calculate_node_flow_updates(network_instance, node, leg, interval)
            scale_flow_updates_to_fill(network_instance, node, leg)
            update_transmission_flows(network_instance, node, leg, interval)
    return None


@njit(fastmath=FASTMATH)
def set_node_fills_and_surpluses(
    network_instance: Network_InstanceType, transmission_case: unicode_type, interval: int64
) -> None:
    """
    Assign the fill energy for balancing and the surplus energy available at each Node in the
    network for the given time interval. Different transmission cases provide different fill
    and surplus energies. This allows transmission to be completed in a sequence of steps,
    with priority given to transmitting certain energy first (e.g., surplus generation from
    solar and wind is transmitted before flexible generation).

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    transmission_case (unicode_type): String defining the transmission case for the fill and
        surplus calculations.
    interval (int64): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: fill, surplus.
    """
    if transmission_case == "surplus":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t, 0)
            node.surplus = -min(node.netload_t, 0)
    elif transmission_case == "storage_discharge":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t - node.storage_power[interval], 0)
            node.surplus = max(node.discharge_max_t[-1] - node.storage_power[interval], 0)

    elif transmission_case == "flexible":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
            node.surplus = max(node.flexible_max_t[-1] - node.flexible_power[interval], 0)

    elif transmission_case == "storage_charge":
        for node in network_instance.nodes.values():
            node.fill = max(node.charge_max_t[-1] + node.storage_power[interval], 0)
            node.surplus = -min(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0.0)

    elif transmission_case == "precharging_surplus":
        for node in network_instance.nodes.values():
            node.fill = node.precharge_fill
            node.surplus = node.existing_surplus

    elif transmission_case == "precharging_transfers":
        for node in network_instance.nodes.values():
            node.fill = node.precharge_fill
            node.surplus = node.precharge_surplus

    elif transmission_case == "precharging_adjust_storage":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t - max(node.storage_power[interval], 0), 0)
            node.surplus = -min(node.netload_t - max(node.storage_power[interval], 0), 0)

    elif transmission_case == "precharging_adjust_surplus":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t - max(node.storage_power[interval], 0.0) - node.flexible_power[interval], 0)
            node.surplus = -min(
                node.netload_t - max(node.storage_power[interval], 0.0) - node.flexible_power[interval], 0
            )
    elif transmission_case == "precharging_adjust_flexible":
        for node in network_instance.nodes.values():
            node.fill = -min(node.storage_power[interval], 0.0)
            node.surplus = -min(
                node.netload_t - max(node.storage_power[interval], 0.0) - node.flexible_power[interval], 0
            )
    else:
        raise RuntimeError("set_node_fills_and_surpluses contains incorrect transmission_case argument.")
    return None


@njit(fastmath=FASTMATH)
def calculate_spillage_and_deficit(network_instance: Network_InstanceType, interval: int64) -> None:
    """
    Calculate the power spilled/curtailed and the unserved power (deficit) for each Node in the network
    upon completing the balancing actions for a time interval.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    interval (int64): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: deficits, spillage.
    """
    for node in network_instance.nodes.values():
        node.deficits[interval] = max(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
        node.spillage[interval] = min(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
    return None


@njit(fastmath=FASTMATH)
def assign_storage_merit_orders(
    network_instance: Network_InstanceType, storages_typed_dict  # DictType(int64, Storage_InstanceType)
) -> None:
    """
    For each Node, finds all Storage instances at that Node. Sorts the Storage systems at that Node from
    shortest to longest duration. Builds a storage merit order array for that Node where the values are
    the Storage.order integers of the sorted Storage instances. That is, it is assumed that short-duration
    storage systems are dispatched first, with long-duration/seasonal storage dispatched last.

    Note that when balancing a deficit block in reverse time, the merit order is reversed.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    storages_typed_dict (DictType(int64, Storage_InstanceType)): Typed dictionary of Storage instances within
        the scenario, keyed by Storage.order.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: storage_merit_order.
    """
    for node in network_instance.nodes.values():
        node_m.assign_storage_merit_order(node, storages_typed_dict)
    return None


@njit(fastmath=FASTMATH)
def assign_flexible_merit_orders(
    network_instance: Network_InstanceType, generators_typed_dict  # DictType(int64, Generators_InstanceType)
) -> None:
    """
    For each Node, finds all flexible Generator instances at that Node. Sorts the flexible Generators at that Node from
    cheapest to most expensive marginal variable costs. Builds a flexible merit order array for that Node where the values are
    the Generator.order integers of the sorted Generator instances. That is, it is assumed that cheap flexible Generators
    are dispatched first, with expensive peaking Generators dispatched last.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    generators_typed_dict (DictType(int64, Generators_InstanceType)): Typed dictionary of Generator
        instances within the scenario, keyed by Generator.order.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: flexible_merit_order.
    """
    for node in network_instance.nodes.values():
        node_m.assign_flexible_merit_order(node, generators_typed_dict)
    return None


@njit(fastmath=FASTMATH)
def calculate_lt_flows(network_instance: Network_InstanceType, interval_resolutions: float64[:]) -> None:
    """
    After completing unit committment, calculates the total long-term major Line flows across the entire
    modelling horizon. Major lines are those that form the transmission network topology and excludes the
    minor lines connecting assets to the transmission network.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    interval_resolutions (float64[:]): A 1-dimensional array containing the resolution for every time interval
        in the unit committment formulation (hours per time interval). An array is used instead of a single
        scalar value to allow for variable time step simplified balancing methods to be developed in future.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Line in Network.major_lines: lt_flows.
    """
    for line in network_instance.major_lines.values():
        line_m.calculate_lt_flow(line, interval_resolutions)
    return None


@njit(fastmath=FASTMATH)
def calculate_lt_line_losses(network_instance: Network_InstanceType) -> float64:
    """
    Estimates the total long-term transmission losses over the modelling horizon for both major and minor
    Lines. Assumes a simplified linear loss factor applied to the line flows.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    float64: Total power losses [GW] across all Lines over the entire modelling horizon.
    """
    total_line_losses = 0.0
    for line in network_instance.major_lines.values():
        total_line_losses += line_m.get_lt_losses(line)
    for line in network_instance.minor_lines.values():
        total_line_losses += line_m.get_lt_losses(line)
    return total_line_losses


@njit(fastmath=FASTMATH)
def reset_flexible(network_instance: Network_InstanceType, interval: int64) -> None:
    """
    Resets the nodal flexible power to zero at each Node for a given time interval.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    interval (int64): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: flexible_power.
    """
    for node in network_instance.nodes.values():
        node.flexible_power[interval] = 0.0
    return None


@njit(fastmath=FASTMATH)
def reset_dispatch(network_instance: Network_InstanceType, interval: int64) -> None:
    """
    Resets the nodal storage power and flexible power to zero at each Node for a given time interval.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    interval (int64): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: flexible_power, storage_power.
    """
    for node in network_instance.nodes.values():
        node.storage_power[interval] = 0.0
        node.flexible_power[interval] = 0.0
    return None


@njit(fastmath=FASTMATH)
def check_precharging_end(network_instance: Network_InstanceType, interval: int64) -> boolean:
    """
    Check whether the start of the deficit block has been reached. While balancing a deficit block,
    the unit committment rules are iterating in reverse time (decreasing intervals). Therefore, the
    start of the deficit block is either the first interval in the optimisation horizon (since there
    are no more intervals to iterate backwards through), or an interval immediately before a deficit
    block interval that has no unserved energy at any Node.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    interval (int64): Index for the time interval.

    Returns:
    -------
    boolean: True if the start of the deficit block has been reached, False if the current time interval
    is not the first interval of the deficit block.
    """
    if interval == 0:
        return True
    for node in network_instance.nodes.values():
        if (
            node.residual_load[interval - 1]
            - node.imports_exports[interval - 1]
            - node.storage_power[interval - 1]
            - node.flexible_power[interval - 1]
            > TOLERANCE
        ):
            return False
    return True


@njit(fastmath=FASTMATH)
def check_existing_surplus(network_instance: Network_InstanceType) -> boolean:
    """
    During the precharging period, check if any Nodes have an existing surplus that can be used to
    precharge Storage systems. Note that the existing surplus is a temporary variable that gets
    updated each time interval during the precharging period.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    boolean: True if any Node has an existing surplus, otherwise False.
    """
    for node in network_instance.nodes.values():
        if node.existing_surplus > TOLERANCE:
            return True
    return False


@njit(fastmath=FASTMATH)
def set_storage_precharge_fills_and_surpluses(network_instance: Network_InstanceType) -> None:
    """
    Updates the temporary precharging period fill and surplus values that are used to evaluate
    inter-storage transfers. The precharge fill and precharge surplus values are used to perform
    intra-node inter-storage transfers, and then initialise the fill and surplus values used for
    inter-storage transfers via transmission.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: precharge_fill, precharge_surplus.
    """
    for node in network_instance.nodes.values():
        node.precharge_fill = node.charge_max_t[-1]
        node.precharge_surplus = node.discharge_max_t[-1]
    return None


@njit(fastmath=FASTMATH)
def set_flexible_precharge_fills_and_surpluses(network_instance: Network_InstanceType) -> None:
    """
    Updates the temporary precharging period fill and surplus values that are used to evaluate
    flexible precharging of Storage systems. The precharge fill and precharge surplus values are used to perform
    intra-node flexible precharging, and then initialise the fill and surplus values used for
    flexible precharging via transmission.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: precharge_fill, precharge_surplus.
    """
    for node in network_instance.nodes.values():
        node.precharge_fill = node.charge_max_t[-1]
        node.precharge_surplus = node.flexible_max_t[-1]
    return None


@njit(fastmath=FASTMATH)
def update_imports_exports_temp(network_instance: Network_InstanceType, interval: int64) -> None:
    """
    Within the precharging period, imports/exports at each Node are adjusted to allow for additional
    transmission of electricity to precharge Storage systems. This function saves the difference between
    the original imports/exports (stored as a temporary value) and the adjusted imports/exports, then
    sets the temporary imports/exports to equal the new adjusted imports/exports.

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.
    interval (int64): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: imports_exports_update, imports_exports_temp.
    """
    for node in network_instance.nodes.values():
        node.imports_exports_update = node.imports_exports_temp - node.imports_exports[interval]
        node_m.set_imports_exports_temp(node, interval)
    return None


@njit(fastmath=FASTMATH)
def check_precharge_fill(network_instance: Network_InstanceType) -> boolean:
    """
    Within the precharging period, checks whether any Node in the Network has fill energy (i.e., Storage prechargers that
    are attempting to be trickle charged).

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    boolean: True if any Node has precharge fill energy it is attempting to balance, otherwise False.
    """
    for node in network_instance.nodes.values():
        if node.precharge_fill > TOLERANCE:
            return True
    return False


@njit(fastmath=FASTMATH)
def check_precharge_surplus(network_instance: Network_InstanceType) -> boolean:
    """
    Within the precharging period, checks whether any Node in the Network has surplus energy (i.e., Storage or flexible
    Generators that are available for trickle charging the prechargers).

    Parameters:
    -------
    network_instance (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    boolean: True if any Node has precharge surplus energy available for trickle charging, otherwise False.
    """
    for node in network_instance.nodes.values():
        if node.precharge_surplus > TOLERANCE:
            return True
    return False
