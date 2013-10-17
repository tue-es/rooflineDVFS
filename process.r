#!/usr/bin/Rscript

# R options
options("width"=220)

##################################
## Functions
##################################

# Functions to concatenate two strings
p <- function(a, b) { return(paste(a, b, sep="")) }
d <- function(a, b) { return(paste(a, b, sep="/")) }

# Function to split a string
ssplit <- function(string, separator) { return(unlist(strsplit(string, separator))) }
fsplit <- function(string, separator) { return(unlist(strsplit(string, separator, fixed=T))) }

# Function to open, read, and close a file
parse <- function(filename) {
	connection = file(filename,"rt")
	lines = readLines(connection)
	close(connection)
	return(lines)
}

# Function to process a file and extract a value corresponding to a string
process <- function(lines, string) {
	results = c()
	for (line in lines) {
		data = ssplit(line, " ")
		if (length(data) > 0 && data[1] == string) {
			results = c(results, as.numeric(tail(data, n=1)))
		}
	}
	return(results)
}

##################################
## GPU parameters
##################################

# Set the voltage scalings for different core and memory clocks
factors_core = c("600"  = 0.7^2,
                 "800"  = 0.8^2,
                 "1000" = 0.9^2,
                 "1200" = 1.0^2)
factors_mem = c("450" = 0.7^2,
                "600" = 0.8^2,
                "750" = 0.9^2,
                "900" = 1.0^2)

# Set the constant power
leakagepower = 55 # 59W * 14/15 = 55W
constantdynamicpower = 10 # From GPUWattch

# Set the default clocks (GTX470 rounded)
defaultcore = 1200
defaultmem = 900

# Set the memory request size (bytes)
requestsize = 128

##################################
## Settings
##################################

# Exclude some benchmarks from the results
excludes = c("")

##################################
## First part of the script (read the data)
##################################

# Initialise the vectors
benchnames = c(); benchtypes = c(); settings1 = c(); settings2 = c(); ids = c()
cycles = c(); dynpower = c(); mempower = c()
instructions = c(); memory_r_ins = c(); memory_w_ins = c(); memory_r_bytes = c(); memory_w_bytes = c()

# Iterate over all the result files
filenames = Sys.glob("simulation_data/*/results*/*/*.txt")
for (filename in filenames) {
	filenamesplit = ssplit(filename, "/")
	powerfile = Sys.glob(d(d(d(d(filenamesplit[1], filenamesplit[2]), filenamesplit[3]), filenamesplit[4]), "gpgpusim_power_report__*.log"))[1]
	detailsfile = p(fsplit(filename, ".txt")[1],".out")
	
	# Extract settings and benchmark name
	benchname = filenamesplit[4]
	if (benchname == "bwbench") benchname = "BW-mb"
	if (benchname == "pebench") benchname = "PE-mb"
	benchtype = filenamesplit[2]
	if (benchtype == "microbenchmarks") benchtype = 0
	if (benchtype == "benchmarks")      benchtype = 1
	setting1  = as.numeric(fsplit(filenamesplit[3], ".")[2])
	setting2  = as.numeric(fsplit(filenamesplit[3], ".")[3])
	
	# Next if we want to include this benchmark
	continue = TRUE
	for (exclude in excludes) {
		if (benchname == exclude) {
			continue = FALSE
		}
	}
	if (continue) {
	
		# Parse the simulation file and the power file
		lines1 = parse(filename)
		lines2 = parse(powerfile)
		
		# Gather information from the data
		val_cycles = process(lines1, "gpu_sim_cycle")
		val_dynpower = process(lines2, "gpu_avg_power")
		val_mempower = process(lines2, "gpu_avg_DRAMP,")
		for (k in 1:length(val_cycles)) {
		
			# Store the results
			benchnames = c(benchnames, benchname)
			benchtypes = c(benchtypes, benchtype)
			settings1  = c(settings1, setting1)
			settings2  = c(settings2, setting2)
			ids = c(ids, k)
			cycles = c(cycles, val_cycles[k])
			dynpower = c(dynpower, val_dynpower[k])
			mempower = c(mempower, val_mempower[k])
		}
		
		# Parse the full-details file as well (to extract data from the nominal case)
		if (setting1==defaultcore && setting2==defaultmem) {
			lines3 = parse(detailsfile)
			val_instructions = process(lines3, "gpgpu_n_tot_thrd_icount")
			val_memory_r_ins = process(lines3, "gpgpu_n_load_insn") / 2 # TODO: Why is this factor 2 needed here?
			val_memory_w_ins = process(lines3, "gpgpu_n_store_insn") / 2 # TODO: Why is this factor 2 needed here?
			val_memory_r_bytes = process(lines3, "gpgpu_n_mem_read_global") * requestsize
			val_memory_w_bytes = process(lines3, "gpgpu_n_mem_write_global") * requestsize
			for (k in 1:length(val_instructions)) {
				instructions = c(instructions, val_instructions[k])
				memory_r_ins = c(memory_r_ins, val_memory_r_ins[k])
				memory_w_ins = c(memory_w_ins, val_memory_w_ins[k])
				memory_r_bytes = c(memory_r_bytes, val_memory_r_bytes[k])
				memory_w_bytes = c(memory_w_bytes, val_memory_w_bytes[k])
			}
		}
	}
}

# Compute the computation intensities (for the roofline model)
int_ins = round(instructions/(memory_r_ins+memory_w_ins), 2)
int_bytes = round(instructions/(memory_r_bytes+memory_w_bytes), 2)

# Gather all the data into a database
db = data.frame(name=benchnames, type=benchtypes, core=settings1, mem=settings2, kID=ids, cycles=cycles,
                totalpower=dynpower+constantdynamicpower+leakagepower, dynpower=dynpower+constantdynamicpower, corepower = (dynpower+constantdynamicpower)-mempower, mempower=mempower, 
                time=as.integer(cycles/settings1))

##################################
## Second part of the script (process the data)
##################################

# Prepare the data (normalise the performance and compute the power and energy efficiency)
count = 1
dbtemp = data.frame()
for (name in unique(db$name)) {
	dbn = db[db$name==name,]
	for (kID in unique(dbn$kID)) {
		dbnk = dbn[dbn$kID==kID,]
		default = dbnk[dbnk$core==defaultcore & dbnk$mem==defaultmem,]
		
		# Performance/power/energy
		scaling_core = factors_core[as.character(dbnk$core)]
		scaling_mem = factors_mem[as.character(dbnk$mem)]
		dynpowerscaled = dbnk$corepower*scaling_core + dbnk$mempower*scaling_mem
		dbnk["performance"] = NA
		dbnk$performance = default$time / dbnk$time
		dbnk["normdynpowerDFS"] = NA; dbnk["normdynpowerDVFS"] = NA
		dbnk$normdynpowerDFS  =  dbnk$dynpower / default$dynpower
		dbnk$normdynpowerDVFS = dynpowerscaled / default$dynpower
		dbnk["normtotalpowerDFS"] = NA; dbnk["normtotalpowerDVFS"] = NA
		dbnk$normtotalpowerDFS  =  (dbnk$dynpower + leakagepower) / (default$dynpower + leakagepower)
		dbnk$normtotalpowerDVFS = (dynpowerscaled + leakagepower*scaling_core) / (default$dynpower + leakagepower)
		dbnk["effDFS"] = NA; dbnk["effDVFS"] = NA
		dbnk$effDFS  = dbnk$performance / dbnk$normtotalpowerDFS
		dbnk$effDVFS = dbnk$performance / dbnk$normtotalpowerDVFS
		
		# Computational intensity
		dbnk["int_ins"] = NA
		dbnk$int_ins = rep(int_ins[count], 7)
		dbnk["int_bytes"] = NA
		dbnk$int_bytes = rep(int_bytes[count], 7)
		
		# Merge the data into the final database
		if (nrow(dbtemp) == 0) {
			dbtemp = dbnk
		} else {
			dbtemp = merge(dbtemp, dbnk, all=T)
		}
		count = count + 1
	}
}
db = dbtemp

# Sort and print the data
db = db[with(db, order(name, kID, core, -mem)), ]
print((db), digits=4)

##################################
## Third part of the script (output the database)
##################################

# Output the database to file
db[,-1] <-round(db[,-1],2)
write.csv(db, file="results/database.csv", row.names=F, quote=F)

##################################