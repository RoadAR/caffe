function solver = get_solver(solver_file)
% solver = get_solver(solver_file)
%   Construct a Solver object from solver_file

pSolver = caffe_('get_solver', solver_file);
solver = caffe.Solver(pSolver);

end
