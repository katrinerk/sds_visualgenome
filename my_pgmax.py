import numpy as np
import itertools
import math
import jax

from pgmax import fgraph, fgroup, infer, vgroup, factor

class MyFactorGraphWrapper:
    def __init__(self):
        # store groups of variables for later addition
        self.vargroups = { }
        # for each variable group: how many variables?
        self.vargroup_size = { }
        
        # store groups of factors for later addition,
        # as a list of objects in pgmax' internal format.
        # only used to construct the factor graph
        self.factorgroups = [ ]

        # for each variable: arity, and groupname and index within the group
        self.var_arity = { }
        self.var_groupname_index = { }

        # for each factor: store a pair (variables, configurations)
        # in the order in which they were defined.
        # variables is a list of variables for the factor,
        # configuraiotns is the list of valid variable combinations
        self.myfactorlist = [ ]

        # stored factor graph
        self.fg = None
        
        # stored results of sum-product run
        self.bp_sumprod_result = None

    ###########3
    # methods that add variables and factors
    
    # mqke a variable group and keep track of
    # a name for each variable.
    # also record the variables, so they can be added to the
    # factor graph later
    def add_variable_group(self, groupname, num_vars, num_states):
        new_vars = vgroup.NDVarArray(num_states=num_states, shape=(num_vars,))
        
        # for each variable, record its name and number of states:
        # mapping from a PGMax variable encoding to a variable name
        self._store_vargroup(new_vars, num_vars, num_states, groupname)
        
        return new_vars

    # make a factor and record it, so it can be added to the factor graph later.
    # also keep track of its valid configurations
    # so we can later use to determine energies
    def add_factor(self, variables, factor_configs = None, log_potentials = None):
        
        # determine factor configurations
        if factor_configs is None:
            # if no configurations given, assume all are valid
            factor_configs = self._all_config(variables)

        # determine log potentials
        if log_potentials is None:
            # if no log potentials are given,
            # assume equal probability for all
            log_potentials = self._uniform_logprob(factor_configs)

        # make and store factor
        new_factor = factor.EnumFactor(
            variables = variables,
            factor_configs = factor_configs,
            log_potentials = log_potentials
        )

        self._store_factor(new_factor)

    # make a group of factors with the same configurations and log potentials
    def add_factor_group(self, variable_groups, factor_configs = None, log_potentials = None):
        # determine factor configurations
        if factor_configs is None:
            # if no configurations given, assume all are valid
            factor_configs = self._all_config(variable_groups[0])

        # determine log potentials
        if log_potentials is None:
            # if no log potentials are given,
            # assume equal probability for all
            log_potentials = self._uniform_logprob(factor_configs)

        new_factorgroup = fgroup.EnumFactorGroup(
            variables_for_factors=variable_groups,
            factor_configs = factor_configs,
            log_potentials = log_potentials)

        self._store_factorgroup(new_factorgroup)

    #####
    # finalizing the factor graph
    
    # using the previously defined variables and factors
    # to construct a factor graph,
    def make(self):
        if len(self.vargroups) == 0:
            raise Exception("no variables")
        
        # make factor graph
        self.fg = fgraph.FactorGraph(variable_groups=list(self.vargroups.values()))
        self.fg.add_factors(self.factorgroups)

    ##############3
    # running inference
    # and reading the results

    # run sum-product algorithm.
    # the result is stored and can be later accessed
    # to obtain marginal weights,
    # factor configuration weights
    # and valuation weights using the readoff_ methods
    def marginal_inference(self):
        
        # run loopy belief propagation: sum-product
        bp = infer.BP(self.fg.bp_state, temperature=1.0)
        self.bp_sumprod_result = bp.run_bp(bp.init(), num_iters=100, damping=0.5)

        beliefs = bp.get_beliefs(self.bp_sumprod_result)
        marginals = infer.get_marginals(beliefs)

        return self._deconstruct_inference_for_var(marginals, compact = True)

    # run max-product algorithm
    # to obtain MAP estimate, and return that estimate right away.
    # nothing is stored.
    def map_inference(self):
        bp = infer.BP(self.fg.bp_state, temperature=0.0)
        bp_arrays = bp.run_bp(bp.init(), num_iters=100, damping=0.5)
        beliefs = bp.get_beliefs(bp_arrays)
        map_states = infer.decode_map_states(beliefs)

        return self._deconstruct_inference_for_var(map_states, expecting_float = True)
        #return map_states

    # compute the energy of the given complete variable assignment
    # given as a dictionary variable -> value
    def readoff_valuation(self, target_assignment):
        if self.bp_sumprod_result is None:
            # need to run sum-product belief propagation before reading off results. do this now.
            self.marginal_inference()
        
        # sanity check: target assignment has to contain all variables
        for groupname in self.vargroups.keys():
            if groupname not in target_assignment:
                raise Exception("method readoff_valuation() requires fully specified assignment, all variable groups must occur, missing " + groupname)
            if len(target_assignment[groupname]) != self.vargroup_size[groupname]:
                raise Exception("number of assignments needs to match number of variables, mismatch for " + groupname)
            
        
        log_energy = 0
                
        global_config_counter = 0
        
        for variables, factor_configs in self.myfactorlist:
            
            # map the target values into the order in which
            # they appear in the factor
            target_vals = [ ]
            for v in variables:
                groupname, varindex = self.var_groupname_index[v]
                if groupname not in target_assignment:
                    raise Exception("readoff valuation: need values for all variables, missing " + groupname)
                value = target_assignment[groupname][varindex]
                target_vals.append(value)
                
            # do we have this in our configurations? 
            local_index = self._find_first(factor_configs, target_vals)
            if local_index is None:
                # no matching factor found, overall energy is minus infinity
                # print("not found")
                return None
            # yes, matching configuration found. Add its log energy to our
            # accumulator
            le= self.bp_sumprod_result.log_potentials[ global_config_counter + local_index]
            # print("found, adding", le)
            log_energy += le

            # update global configuration counter to reflect
            # the number of configurations we've seen soon far
            global_config_counter += len(factor_configs)

        return log_energy


    # force particular nodes to have particular values:
    # make a new MyFactorGraphWrapper object that is a copy of hte
    # current one except that new unary factors have been added to
    # force particular nodes to have particular values.
    #
    # if one of the target nodes already has a unary factor attached, it will
    # be replaced by the value-forcing factor.
    # if one of the target nodes is part of a factor group with a unary factor attached,
    # the factor group is be replaced
    # by individual factors, except for the target nodes, which will have unary value-forcing factors
    #
    # input:
    # list of triples (vargroupname, index of target variable within the variable group, value to force)
    def fg_with_evidence(self, evidence):
        # determine target variables and their values
        target_vars_vals = { }
        for groupname, index, value in evidence:
            var = self.vargroups[groupname][index]
            target_vars_vals[ var ] = value

        # make new factor graph object
        new_fg = MyFactorGraphWrapper()
        new_fg.vargroups = self.vargroups
        new_fg.vargroup_size = self.vargroup_size
        new_fg.var_arity = self.var_arity
        new_fg.var_groupname_index = self.var_groupname_index

        # add factors and factor groups:
        # factors and factor groups that we had before, minus ones that would conflict with
        # the value-forcing unary factors
        for ftype, fvariables, fconfigs, flogprobs in self._each_factor_or_factorgroup():
            if ftype == "single":
                if any(fvariables == [t] for t in target_vars_vals.keys()):
                    # unary factor that would conflict with a ne one, skip
                    pass
                else:
                    # keep this factor
                    new_fg.add_factor(fvariables,factor_configs =  fconfigs, log_potentials = flogprobs)
                    
            elif ftype == "group":
                if any([t] in fvariables for t in target_vars_vals.keys()):
                    # this is a factor group with unary factors that would conflict with the new unary factors.
                    # add as individual factors, whenever there's no conflict
                    for varset in fvariables:
                        if not(any(varset == [t] for t in target_vars_vals.keys())):
                            new_fg.add_factor(varset, factor_configs = fconfigs, log_potentials = flogprobs)
                else:
                    new_fg.add_factor_group(fvariables, factor_configs = fconfigs, log_potentials = flogprobs)
            else:
                raise Exception("weird factor type " + ftype)

        # add value-forcing unary factors
        for var, value in target_vars_vals.items():
            new_fg.add_factor( [var], factor_configs = np.array([[value]]), log_potentials = np.array([0.0]))

        # finalize the factor graph
        new_fg.make()

        return new_fg
        
    # set fixed values for some variables.
    # format:
    # group name -> (variable index, value)
    #
    # internal format for evidence:
    # variable object -> value
    def map_inference_with_evidence(self, evidence):
        # internal_evidence = { }
        # for groupname, index_val in evidence.items():
        #     vargroup = self.vargroups[groupname]
        #     for i, val in index_val:
        #         internal_evidence[vargroup[i]] = val

        # print(internal_evidence)

        # bp = infer.BP(self.fg.bp_state, temperature=0.0)
        # bp_arrays = bp.init( evidence_updates=internal_evidence )
        # bp_arrays = bp.run_bp(bp_arrays, num_iters=100, damping=0.5)

        for f in self.factorgroups:
            print(type(f), "variables" in dir(f))
            if "variables" not in dir(f): print(f.variables_for_factors)
        return None
    
        for groupname, index_val in evidence.items():
            vargroup = self.vargroups[groupname]
            for i, val in index_val:
                # if we already have a unary factor for this variable, take it out
                self.factorgroups = [f for f in self.factorgroups if "variables" not in dir(f) or ("variables" in dir(f) and f.variables != [vargroup[i]])]
                self.add_factor([vargroup[i]], factor_configs = np.array([[val]]), log_potentials = np.array([0.0]))


        self.fg = fgraph.FactorGraph(variable_groups=list(self.vargroups.values()))
        self.fg.add_factors(self.factorgroups)
        
        bp = infer.BP(self.fg.bp_state, temperature=0.0)
        bp_arrays = bp.run_bp(bp.init(), num_iters=100, damping=0.5)
        beliefs = bp.get_beliefs(bp_arrays)
        map_states = infer.decode_map_states(beliefs)

        return self._deconstruct_inference_for_var(map_states, expecting_float = True)
            
    #########################################
    # helper methods,
    # internal
    
    # find first occurrence of target in nparray,
    # and return index.
    # None if not found.
    # assumes that target is also an array 
    def _find_first(self, nparray, target):
        for i, val in enumerate(nparray):
            if np.array_equal(val, target): return i
        return None
    
    # given a list of variables,
    # make configurations to state that all combinations of values
    # for these variables are valid
    def _all_config(self, variables):
        if len(variables) == 1:
            v= variables[0]
            return np.arange(self.var_arity[v])[:, None]

        else:
            num_states =  [ self.var_arity[v] for v in variables ]
            return np.array(list(itertools.product(*[np.arange(n) for n in num_states])))


    # given a list of configurations,
    # make an array of log probabilities that are uniform across those
    # configurations
    def _uniform_logprob(self, configurations):
        n = len(configurations)
        if n ==0:
            raise Exception("zero valid configurations, cannot make uniform log probabilities")
        
        return np.log(np.ones(n)/n)

    # store variable group 
    def _store_vargroup(self, vargroup, num_vars, num_states, groupname):

        # store in vargroups so it can be added to the graph 
        self.vargroups[groupname] = vargroup
        self.vargroup_size[groupname] = num_vars

        for i in range(num_vars):
            self.var_arity[vargroup[i]] = num_states
            self.var_groupname_index[ vargroup[i] ] = (groupname, i)

    # store factor
    def _store_factor(self, factor):
        self.factorgroups.append(factor)
        
        self.myfactorlist.append( (factor.variables, factor.factor_configs) )

    # store factor group
    def _store_factorgroup(self, factorgroup):
        self.factorgroups.append(factorgroup)
        
        for variables in factorgroup.variables_for_factors:
            self.myfactorlist.append( (variables, factorgroup.factor_configs) )


    # given an inference result
    # that is a mapping from variable groups to something,
    # make a dictionary that maps individual variable names
    # to their appropriate value
    def _deconstruct_inference_for_var(self, infresult, expecting_float = False, compact = False):
        
        # infresult is a dictionary
        # in which the keys are variable groups
        # and the values are sequences of either MAP values
        # or arrays of marginals
        # for the variables in that group
        #
        # self.vargroups has the variable groups in the order in which they were defined
        # make readable output: variable name -> value
        retv = { }
        for groupname, vargroup in self.vargroups.items():
            thisresult = infresult[vargroup]
            if expecting_float:
                retv[groupname] = [thisresult[i].item() for i in range(len(thisresult))]
            else:
                if compact:
                    retv[groupname] = [ ]
                    for i in range(len(thisresult)):
                        arr = jax.device_get(thisresult[i])
                        nonzero_indices = [i.item() for i in np.nonzero(arr)[0]]
                        # dictionary: value index -> probability,
                        # but only where the probability is nonzero
                        retv[groupname].append( dict( zip([i.item() for i in np.nonzero(arr)[0]], [v.item() for v in arr[np.nonzero(arr)]])))
                else:
                    retv[groupname] = [list(jax.device_get(thisresult[i])) for i in range(len(thisresult)) ]
                
        return retv    
            

    # for each factor or factor group in self.factorgroups:
    # return tuple
    # ("single" or "group",
    #   list of variables for single or list of lists of variables for group,
    #   valid factors,
    #   log probabilities
    def _each_factor_or_factorgroup(self):
        for f in self.factorgroups:
            if "variables" in dir(f):
                # individual factor
                yield ("single", f.variables, f.factor_configs, f.log_potentials)
            else:
                yield ("group", f.variables_for_factors, f.factor_configs, f.log_potentials)
