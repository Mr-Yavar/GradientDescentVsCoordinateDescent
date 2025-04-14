using System;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using GDvsCD.Services;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.AspNetCore.Mvc;

namespace GDvsCD.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class CalcController : ControllerBase
    {
        private readonly ILogger<CalcController> _logger;

        public CalcController(ILogger<CalcController> logger)
        {
            _logger = logger;
        }

        [HttpGet("[action]")]
        public async Task<IActionResult> Get(float eta, int numIter)
        {
            // Create an ILGPU context.
            using var context = Context.CreateDefault();
            // Get the first available accelerator (e.g. a Cuda or OpenCL GPU).
            using var accelerator = context
                .GetPreferredDevice(preferCPU: false)
                .CreateAccelerator(context);

            // Read the JSON file content.
            string json = await System.IO.File.ReadAllTextAsync("./Dataset/features.json");

            var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };

            double[,] X = Algo.ConvertJaggedTo2D(
                JsonSerializer.Deserialize<double[][]>(json, options)
            );

            json = await System.IO.File.ReadAllTextAsync("./Dataset/output.json");

            double[] y = JsonSerializer.Deserialize<double[]>(json, options);
            // y is the target vector.

            // Initial guess for theta (n parameters); here two parameters.
            double[] theta1 = new double[]
            {
            4,2,3,6,5,4
          
            };
            double[] theta2 = new double[]
            {

               4,2,3,6,5,4

            };

            double cdTime, gdTime;

        
       
     
         
            var watch = System.Diagnostics.Stopwatch.StartNew();
            // Run coordinate descent.
            var resultCD = Algo.CoordinateDescent(
                theta1,
                X,
                y,
                accelerator,
                alpha: eta,
                numIters: numIter
            );
            watch.Stop();
        
            cdTime= watch.ElapsedMilliseconds;
            watch.Reset();
            watch.Start();
            var resultGD = Algo.GradientDescent(
                theta2,
                X,
                y,
                accelerator,
                alpha: eta,
                numIters: numIter
            );
            watch.Stop();

            gdTime = watch.ElapsedMilliseconds;

            // Dispose of the accelerator (automatically done via using-statement).
            accelerator.Dispose();

            return Ok(
                new
                {
                    CD = new { costHistory = resultCD.costHistory,timing=cdTime},
                    GD = new { costHistory = resultGD.costHistory,timing=gdTime},
                }
            );
        }
    }
}
